# SPDX-License-Identifier: AGPL-3.0-or-later
"""
OpenADR 2.0b Virtual End Node (VEN) Client
============================================
Receives grid signals from a Virtual Top Node (VTN) and translates them
into Energivanu MPC + scheduler actions.

OpenADR (Open Automated Demand Response) is the standard protocol for
utility-to-customer demand response communication. Used by ERCOT, CAISO,
and utilities worldwide.

Signal Levels:
    - SIMPLE: Simple signal with 4 levels (0-3)
    - PAYLOAD: Payload signal with specific kW targets
    - PRICE: Price-based signals (not implemented)

Usage::

    from energivanu.grid import OpenADRVEN, GridEvent, GridSignalLevel

    # Create VEN
    ven = OpenADRVEN(
        vtn_url="https://vtn.example.com/OpenADR2/Simple/2.0b",
        ven_id="energivanu-001",
    )

    # Register callbacks
    def on_event(event: GridEvent):
        print(f"Grid signal: {event.signal_level}, action: {event.action}")

    ven.on_event_callback = on_event

    # Start listening (blocking)
    ven.start()

    # Or process events manually
    ven.poll_events()

Dependencies::

    # No external dependencies — uses built-in HTTP + XML
    # For production, install: pip install openadr
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
from xml.etree import ElementTree as ET

from ..logging_config import get_logger, timed

logger = get_logger("grid")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenADR 2.0b namespaces
OADR_NS = {
    "oadr": "http://openadr.org/oadr-2.0b",
    "ei": "http://docs.oasis-open.org/ns/energyinterop/201110",
    "xcal": "http://docs.oasis-open.org/ns/calendar/201110",
    "strm": "http://docs.oasis-open.org/ns/energyinterop/201110/payloads",
}

class GridSignalLevel(IntEnum):
    """OpenADR simple signal levels."""
    NORMAL = 0      # No curtailment needed
    MODERATE = 1    # Minor reduction (10-20%)
    HIGH = 2        # Significant reduction (20-50%)
    CRITICAL = 3    # Emergency curtailment (50%+)


# Signal level mappings (must be after GridSignalLevel definition)
SIGNAL_LEVEL_MAP = {
    "0": GridSignalLevel.NORMAL,
    "1": GridSignalLevel.MODERATE,
    "2": GridSignalLevel.HIGH,
    "3": GridSignalLevel.CRITICAL,
}


@dataclass
class GridEvent:
    """A parsed OpenADR grid event."""
    event_id: str
    signal_level: GridSignalLevel
    signal_value: float           # Target value (kW reduction or price)
    start_time: datetime
    end_time: datetime
    priority: int                 # Lower = higher priority
    action: str                   # Human-readable action
    raw_payload: Optional[Dict] = None

    @property
    def is_active(self) -> bool:
        """Check if event is currently active."""
        now = datetime.now(timezone.utc)
        return self.start_time <= now <= self.end_time

    @property
    def duration_seconds(self) -> float:
        """Event duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "signal_level": self.signal_level.name,
            "signal_value": self.signal_value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "priority": self.priority,
            "action": self.action,
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Signal Action Mapping
# ---------------------------------------------------------------------------

SIGNAL_ACTIONS = {
    GridSignalLevel.NORMAL: {
        "action": "none",
        "power_reduction_pct": 0,
        "bess_action": "hold",
        "scheduler_action": "normal",
    },
    GridSignalLevel.MODERATE: {
        "action": "reduce_10pct",
        "power_reduction_pct": 10,
        "bess_action": "discharge_moderate",
        "scheduler_action": "stagger_phases",
    },
    GridSignalLevel.HIGH: {
        "action": "reduce_30pct",
        "power_reduction_pct": 30,
        "bess_action": "discharge_high",
        "scheduler_action": "stagger_aggressive",
    },
    GridSignalLevel.CRITICAL: {
        "action": "reduce_50pct_plus",
        "power_reduction_pct": 50,
        "bess_action": "discharge_max",
        "scheduler_action": "pause_non_critical",
    },
}


# ---------------------------------------------------------------------------
# OpenADR VEN Client
# ---------------------------------------------------------------------------

class OpenADRVEN:
    """
    OpenADR 2.0b Virtual End Node client.

    Receives demand response signals from a VTN (Virtual Top Node)
    and translates them into Energivanu actions.

    Supports:
    - SIMPLE signals (4 levels: 0-3)
    - PAYLOAD signals (specific kW targets)
    - Event polling and push notifications

    Args:
        vtn_url: URL of the VTN (Virtual Top Node) endpoint.
        ven_id: This VEN's identifier.
        poll_interval_s: Seconds between event polls.
        auth_token: Bearer token for VTN authentication.
        mpc_controller: MPCController instance for battery dispatch.
        scheduler: PhaseStaggeringScheduler for phase control.
    """

    def __init__(
        self,
        vtn_url: str = "http://localhost:8080/OpenADR2/Simple/2.0b",
        ven_id: str = "energivanu-001",
        poll_interval_s: float = 30.0,
        auth_token: Optional[str] = None,
        mpc_controller: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        self.vtn_url = vtn_url
        self.ven_id = ven_id
        self.poll_interval_s = poll_interval_s
        self.auth_token = auth_token
        self.mpc = mpc_controller
        self.scheduler = scheduler

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: List[GridEvent] = []
        self._active_event: Optional[GridEvent] = None
        self._event_history: List[GridEvent] = []
        self._poll_count = 0
        self._error_count = 0

        # Callbacks
        self.on_event_callback: Optional[Callable[[GridEvent], None]] = None
        self.on_signal_change: Optional[Callable[[GridSignalLevel], None]] = None

        logger.info(
            "OpenADR VEN initialized",
            extra={"vtn_url": vtn_url, "ven_id": ven_id},
        )

    # ------------------------------------------------------------------
    # Event Processing
    # ------------------------------------------------------------------

    @timed("grid.openadr.process_event")
    def process_event(self, event: GridEvent) -> Dict[str, Any]:
        """
        Process a grid event and generate MPC + scheduler commands.

        Args:
            event: The grid event to process.

        Returns:
            Dictionary with recommended actions.
        """
        self._active_event = event
        self._event_history.append(event)

        # Get action mapping
        action = SIGNAL_ACTIONS.get(event.signal_level, SIGNAL_ACTIONS[GridSignalLevel.NORMAL])

        result = {
            "event": event.to_dict(),
            "recommended_actions": action,
            "mpc_command": None,
            "scheduler_command": None,
        }

        # Generate MPC command if controller available
        if self.mpc is not None:
            target_reduction_pct = action["power_reduction_pct"]
            if target_reduction_pct > 0:
                # Tell MPC to reduce grid power by this percentage
                result["mpc_command"] = {
                    "action": "reduce_power",
                    "reduction_pct": target_reduction_pct,
                    "bess_action": action["bess_action"],
                }
                logger.info(
                    "MPC command generated",
                    extra={
                        "signal_level": event.signal_level.name,
                        "reduction_pct": target_reduction_pct,
                    },
                )

        # Generate scheduler command if available
        if self.scheduler is not None:
            sched_action = action["scheduler_action"]
            if sched_action != "normal":
                result["scheduler_command"] = {
                    "action": sched_action,
                    "signal_level": event.signal_level.name,
                }
                logger.info(
                    "Scheduler command generated",
                    extra={"action": sched_action},
                )

        # Fire callback
        if self.on_event_callback:
            try:
                self.on_event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Fire signal change callback
        if self.on_signal_change:
            try:
                self.on_signal_change(event.signal_level)
            except Exception as e:
                logger.error(f"Signal change callback error: {e}")

        return result

    def get_current_signal_level(self) -> GridSignalLevel:
        """Get the current active signal level."""
        if self._active_event and self._active_event.is_active:
            return self._active_event.signal_level
        return GridSignalLevel.NORMAL

    def get_active_event(self) -> Optional[GridEvent]:
        """Get the currently active event, or None."""
        if self._active_event and self._active_event.is_active:
            return self._active_event
        return None

    # ------------------------------------------------------------------
    # Event Polling
    # ------------------------------------------------------------------

    @timed("grid.openadr.poll")
    def poll_events(self) -> List[GridEvent]:
        """
        Poll VTN for new events.

        Returns:
            List of new GridEvents received.
        """
        try:
            url = f"{self.vtn_url}/EiEvent?venID={self.ven_id}"
            headers = {"Accept": "application/xml"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read().decode("utf-8")

            events = self._parse_events_xml(xml_data)
            self._poll_count += 1

            if events:
                logger.info(
                    "received events from VTN",
                    extra={"count": len(events), "poll": self._poll_count},
                )

            return events

        except urllib.error.URLError as e:
            self._error_count += 1
            logger.warning(f"VTN poll failed: {e}")
            return []
        except Exception as e:
            self._error_count += 1
            logger.error(f"Event poll error: {e}")
            return []

    def _parse_events_xml(self, xml_data: str) -> List[GridEvent]:
        """Parse OpenADR event XML into GridEvent objects."""
        events = []
        try:
            root = ET.fromstring(xml_data)

            for event_elem in root.findall(".//oadr:oadrEvent", OADR_NS):
                try:
                    event = self._parse_single_event(event_elem)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event: {e}")

        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")

        return events

    def _parse_single_event(self, event_elem: ET.Element) -> Optional[GridEvent]:
        """Parse a single oadrEvent element."""
        # Extract event ID
        event_id_elem = event_elem.find(".//ei:eventID", OADR_NS)
        event_id = event_id_elem.text if event_id_elem is not None else "unknown"

        # Extract signal
        signal_elem = event_elem.find(".//ei:signalPayload", OADR_NS)
        if signal_elem is None:
            signal_elem = event_elem.find(".//ei:currentValue", OADR_NS)

        signal_value = 0.0
        if signal_elem is not None and signal_elem.text:
            try:
                signal_value = float(signal_elem.text)
            except ValueError:
                pass

        # Map to signal level
        signal_level = SIGNAL_LEVEL_MAP.get(str(int(signal_value)), GridSignalLevel.NORMAL)

        # Extract times
        start_elem = event_elem.find(".//xcal:dtstart/xcal:datetime", OADR_NS)
        end_elem = event_elem.find(".//xcal:dtend/xcal:datetime", OADR_NS)

        now = datetime.now(timezone.utc)
        start_time = now
        end_time = now

        if start_elem is not None and start_elem.text:
            try:
                start_time = datetime.fromisoformat(start_elem.text.replace("Z", "+00:00"))
            except ValueError:
                pass
        if end_elem is not None and end_elem.text:
            try:
                end_time = datetime.fromisoformat(end_elem.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Extract priority
        priority_elem = event_elem.find(".//ei:priority", OADR_NS)
        priority = (
            int(priority_elem.text)
            if priority_elem is not None and priority_elem.text
            else 0
        )

        # Generate action description
        action_info = SIGNAL_ACTIONS.get(signal_level, SIGNAL_ACTIONS[GridSignalLevel.NORMAL])
        action = action_info["action"]

        return GridEvent(
            event_id=event_id,
            signal_level=signal_level,
            signal_value=signal_value,
            start_time=start_time,
            end_time=end_time,
            priority=priority,
            action=action,
        )

    # ------------------------------------------------------------------
    # Background Polling
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background event polling."""
        if self._running:
            logger.warning("VEN already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("OpenADR VEN polling started")

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("OpenADR VEN stopped")

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            events = self.poll_events()
            for event in events:
                self.process_event(event)
            time.sleep(self.poll_interval_s)

    # ------------------------------------------------------------------
    # Mock / Testing
    # ------------------------------------------------------------------

    def simulate_event(
        self,
        level: GridSignalLevel = GridSignalLevel.MODERATE,
        duration_seconds: float = 300.0,
        signal_value: Optional[float] = None,
    ) -> GridEvent:
        """
        Simulate a grid event for testing (no VTN needed).

        Args:
            level: Signal level to simulate.
            duration_seconds: Event duration.
            signal_value: Override signal value.

        Returns:
            The simulated GridEvent.
        """
        now = datetime.now(timezone.utc)
        if signal_value is None:
            signal_value = float(level)

        event = GridEvent(
            event_id=f"sim_{int(time.time())}",
            signal_level=level,
            signal_value=signal_value,
            start_time=now,
            end_time=now + __import__("datetime").timedelta(seconds=duration_seconds),
            priority=level,
            action=SIGNAL_ACTIONS[level]["action"],
            raw_payload={"simulated": True},
        )

        self.process_event(event)
        logger.info(
            "simulated grid event",
            extra={"level": level.name, "duration_s": duration_seconds},
        )
        return event

    def get_stats(self) -> Dict[str, Any]:
        """Return VEN statistics."""
        return {
            "running": self._running,
            "ven_id": self.ven_id,
            "vtn_url": self.vtn_url,
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "total_events": len(self._event_history),
            "active_event": self._active_event.to_dict() if self._active_event else None,
            "current_signal_level": self.get_current_signal_level().name,
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_ven_from_config(config: Any = None) -> OpenADRVEN:
    """Create an OpenADRVEN from EnergivanuConfig."""
    if config is None:
        from ..config import get_config
        config = get_config()

    return OpenADRVEN(
        vtn_url="http://localhost:8080/OpenADR2/Simple/2.0b",
        ven_id="energivanu-001",
    )
