# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BESS Modbus Mock Server
========================
Simulates a Tesla Megapack Modbus TCP interface for testing MPC controllers.
Provides standard register map for SOC, power, status, and temperature.

Usage::

    from energivanu.bess import BESSModbusServer

    server = BESSModbusServer(port=5020)
    server.set_soc(0.75)        # Set initial SOC
    server.set_power(50.0)      # Set current power output (MW)
    server.start()              # Start serving (blocking)

    # In another process, connect with Modbus client:
    from pymodbus.client import ModbusTcpClient
    client = ModbusTcpClient("localhost", port=5020)
    soc = client.read_holding_registers(100, 1)  # Read SOC register

Dependencies::

    pip install pymodbus   # Modbus TCP server/client
    # Falls back to simple HTTP API if pymodbus not installed
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from ..logging_config import get_logger

logger = get_logger("bess")

# ---------------------------------------------------------------------------
# Try importing pymodbus
# ---------------------------------------------------------------------------

_PYMODBUS_AVAILABLE = False
try:
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
        ModbusSlaveContext,
    )
    from pymodbus.server import StartTcpServer
    _PYMODBUS_AVAILABLE = True
    logger.info("pymodbus available — using Modbus TCP server")
except ImportError:
    logger.warning(
        "pymodbus not installed — using HTTP API fallback. "
        "Install with: pip install pymodbus"
    )


# ---------------------------------------------------------------------------
# Register Map (Tesla Megapack compatible)
# ---------------------------------------------------------------------------

@dataclass
class ModbusRegisterMap:
    """
    Standard BESS Modbus register map.

    Maps register addresses to battery parameters.
    Compatible with Tesla Megapack Modbus interface.
    """
    # Register addresses
    SOC_REGISTER: int = 100          # State of Charge (%) × 100
    POWER_REGISTER: int = 102        # Current power (kW), signed
    STATUS_REGISTER: int = 104       # Status word
    VOLTAGE_REGISTER: int = 106      # Terminal voltage (V) × 10
    CURRENT_REGISTER: int = 108      # Current (A) × 10, signed
    TEMPERATURE_REGISTER: int = 110  # Temperature (°C) × 10
    MAX_CHARGE_REGISTER: int = 112   # Max charge power (kW)
    MAX_DISCHARGE_REGISTER: int = 114 # Max discharge power (kW)
    FAULT_REGISTER: int = 116        # Fault code
    CYCLE_COUNT_REGISTER: int = 118  # Cycle count × 100

    # Status word bits
    STATUS_STANDBY: int = 0x0001
    STATUS_CHARGING: int = 0x0002
    STATUS_DISCHARGING: int = 0x0004
    STATUS_FAULT: int = 0x0008
    STATUS_GRID_CONNECTED: int = 0x0010
    STATUS_BMS_READY: int = 0x0020


# ---------------------------------------------------------------------------
# BESS State
# ---------------------------------------------------------------------------

@dataclass
class BESSState:
    """Internal BESS state for the mock server."""
    soc_pct: float = 50.0           # State of Charge (%)
    power_kw: float = 0.0           # Current power (kW), + = discharge
    voltage_v: float = 1200.0       # Terminal voltage (V)
    current_a: float = 0.0          # Current (A)
    temperature_c: float = 25.0     # Temperature (°C)
    status: int = 0x0021            # Standby + BMS Ready
    max_charge_kw: float = 319200.0 # Max charge (kW) = 319.2 MW
    max_discharge_kw: float = 319200.0 # Max discharge (kW)
    fault_code: int = 0             # No fault
    cycle_count: float = 0.0        # Equivalent cycles
    capacity_mwh: float = 655.2     # Total capacity


# ---------------------------------------------------------------------------
# Modbus TCP Server (pymodbus)
# ---------------------------------------------------------------------------

class _ModbusBackend:
    """Modbus data store backend using pymodbus."""

    def __init__(self, state: BESSState, register_map: ModbusRegisterMap):
        self.state = state
        self.rm = register_map
        self._lock = threading.Lock()

        # Initialize register block with zeros
        self._registers = [0] * 200

    def update_registers(self) -> None:
        """Sync BESS state to Modbus registers."""
        with self._lock:
            rm = self.rm
            self._registers[rm.SOC_REGISTER] = int(self.state.soc_pct * 100)
            self._registers[rm.POWER_REGISTER] = int(self.state.power_kw)
            self._registers[rm.STATUS_REGISTER] = self.state.status
            self._registers[rm.VOLTAGE_REGISTER] = int(self.state.voltage_v * 10)
            self._registers[rm.CURRENT_REGISTER] = int(self.state.current_a * 10)
            self._registers[rm.TEMPERATURE_REGISTER] = int(self.state.temperature_c * 10)
            self._registers[rm.MAX_CHARGE_REGISTER] = int(self.state.max_charge_kw)
            self._registers[rm.MAX_DISCHARGE_REGISTER] = int(self.state.max_discharge_kw)
            self._registers[rm.FAULT_REGISTER] = self.state.fault_code
            self._registers[rm.CYCLE_COUNT_REGISTER] = int(self.state.cycle_count * 100)

    def get_register(self, address: int) -> int:
        """Read a register value."""
        with self._lock:
            if 0 <= address < len(self._registers):
                return self._registers[address]
            return 0

    def set_register(self, address: int, value: int) -> None:
        """Write a register value."""
        with self._lock:
            if 0 <= address < len(self._registers):
                self._registers[address] = value


class BESSModbusServer:
    """
    BESS Modbus TCP mock server.

    Simulates a Tesla Megapack Modbus interface for testing MPC controllers.
    Provides standard register map for SOC, power, status, and temperature.

    When pymodbus is not available, starts a simple HTTP API server instead.

    Args:
        host: Server bind address.
        port: Server port (default 5020 to avoid requiring root).
        capacity_mwh: Total battery capacity in MWh.
        max_power_mw: Maximum charge/discharge power in MW.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5020,
        capacity_mwh: float = 655.2,
        max_power_mw: float = 319.2,
    ):
        self.host = host
        self.port = port
        self.state = BESSState(
            capacity_mwh=capacity_mwh,
            max_charge_kw=max_power_mw * 1000,
            max_discharge_kw=max_power_mw * 1000,
        )
        self.register_map = ModbusRegisterMap()
        self._backend = _ModbusBackend(self.state, self.register_map)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # State setters (used by MPC controller)
    # ------------------------------------------------------------------

    def set_soc(self, soc_pct: float) -> None:
        """Set State of Charge (0-100%)."""
        self.state.soc_pct = max(0.0, min(100.0, soc_pct))
        self._backend.update_registers()

    def set_power(self, power_mw: float) -> None:
        """Set current power output in MW. Positive = discharge."""
        self.state.power_kw = power_mw * 1000
        self.state.current_a = (
            power_mw * 1e6 / self.state.voltage_v
            if self.state.voltage_v > 0 else 0
        )
        if power_mw > 0:
            self.state.status = (self.state.status & ~0x0006) | 0x0004  # Discharging
        elif power_mw < 0:
            self.state.status = (self.state.status & ~0x0006) | 0x0002  # Charging
        else:
            self.state.status = (self.state.status & ~0x0006) | 0x0001  # Standby
        self._backend.update_registers()

    def set_temperature(self, temp_c: float) -> None:
        """Set battery temperature."""
        self.state.temperature_c = temp_c
        self._backend.update_registers()

    def set_fault(self, fault_code: int) -> None:
        """Set fault code (0 = no fault)."""
        self.state.fault_code = fault_code
        if fault_code != 0:
            self.state.status |= 0x0008  # Fault bit
        else:
            self.state.status &= ~0x0008
        self._backend.update_registers()

    def get_state_dict(self) -> Dict[str, Any]:
        """Get current BESS state as dictionary."""
        return {
            "soc_pct": round(self.state.soc_pct, 2),
            "power_kw": round(self.state.power_kw, 1),
            "power_mw": round(self.state.power_kw / 1000, 4),
            "voltage_v": round(self.state.voltage_v, 1),
            "current_a": round(self.state.current_a, 1),
            "temperature_c": round(self.state.temperature_c, 1),
            "status": self.state.status,
            "status_str": self._status_string(),
            "max_charge_kw": self.state.max_charge_kw,
            "max_discharge_kw": self.state.max_discharge_kw,
            "fault_code": self.state.fault_code,
            "cycle_count": round(self.state.cycle_count, 2),
            "step_count": self._step_count,
        }

    def _status_string(self) -> str:
        """Convert status word to human-readable string."""
        parts = []
        if self.state.status & 0x0001:
            parts.append("STANDBY")
        if self.state.status & 0x0002:
            parts.append("CHARGING")
        if self.state.status & 0x0004:
            parts.append("DISCHARGING")
        if self.state.status & 0x0008:
            parts.append("FAULT")
        if self.state.status & 0x0010:
            parts.append("GRID_CONNECTED")
        if self.state.status & 0x0020:
            parts.append("BMS_READY")
        return " | ".join(parts) if parts else "UNKNOWN"

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self, blocking: bool = False) -> None:
        """
        Start the BESS Modbus server.

        Args:
            blocking: If True, blocks the calling thread. If False, starts in background.
        """
        if self._running:
            logger.warning("server already running")
            return

        self._backend.update_registers()
        self._running = True

        if blocking:
            self._run_server()
        else:
            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()
            logger.info("BESS server started in background", extra={"port": self.port})

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("BESS server stopped")

    def _run_server(self) -> None:
        """Run the server (blocking)."""
        if _PYMODBUS_AVAILABLE:
            self._run_modbus_server()
        else:
            self._run_http_server()

    def _run_modbus_server(self) -> None:
        """Run pymodbus TCP server."""
        try:
            # Create data store
            block = ModbusSequentialDataBlock(0, self._backend._registers)
            store = ModbusSlaveContext(
                hr=block,  # Holding registers
                di=None,
                co=None,
                ir=None,
                zero_mode=True,
            )
            context = ModbusServerContext(slaves=store, single=True)

            logger.info(
                "starting Modbus TCP server",
                extra={"host": self.host, "port": self.port},
            )

            # This blocks
            StartTcpServer(
                context=context,
                address=(self.host, self.port),
            )
        except Exception as e:
            logger.error(f"Modbus server error: {e}")
            self._running = False

    def _run_http_server(self) -> None:
        """Run HTTP API fallback when pymodbus is not available."""
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._json_response({"status": "ok"})
                elif self.path == "/state":
                    self._json_response(server_ref.get_state_dict())
                elif self.path.startswith("/register/"):
                    try:
                        addr = int(self.path.split("/")[-1])
                        val = server_ref._backend.get_register(addr)
                        self._json_response({"address": addr, "value": val})
                    except ValueError:
                        self._json_response({"error": "invalid address"}, 400)
                else:
                    self._json_response({"endpoints": ["/health", "/state", "/register/<addr>"]})

            def do_POST(self):
                if self.path == "/set_power":
                    content = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(content))
                    server_ref.set_power(body.get("power_mw", 0))
                    self._json_response(server_ref.get_state_dict())
                elif self.path == "/set_soc":
                    content = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(content))
                    server_ref.set_soc(body.get("soc_pct", 50))
                    self._json_response(server_ref.get_state_dict())
                else:
                    self._json_response({"error": "not found"}, 404)

            def _json_response(self, data, code=200):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        try:
            httpd = HTTPServer((self.host, self.port), Handler)
            logger.info(
                "BESS HTTP API server started (pymodbus not available)",
                extra={"host": self.host, "port": self.port},
            )
            while self._running:
                httpd.handle_request()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            self._running = False


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_server_from_config(config: Any = None) -> BESSModbusServer:
    """Create a BESSModbusServer from EnergivanuConfig."""
    if config is None:
        from ..config import get_config
        config = get_config()

    return BESSModbusServer(
        host=config.hardware.modbus_host,
        port=config.hardware.modbus_port,
        capacity_mwh=config.battery.total_capacity_mwh,
        max_power_mw=config.battery.total_power_mw,
    )
