# ENERGIVANU — Battery Signal Classification & Dispatch

## Overview

This document covers battery energy storage systems (BESS) for data centers, signal classification methods, and dispatch optimization strategies. These are the pieces we need to build the SAFE/PREPARE/CRITICAL signal system.

---

## Key Papers Found

### Paper A: Grid-Forming BESS as UPS (Azizi et al., 2026)
**Title:** "Strengthening data center operations using grid-forming battery energy storage as a line-interactive uninterruptible power supply"
**URL:** https://www.sciencedirect.com/science/article/pii/S0142061526000803
**Key insight:** Battery as dual-purpose UPS + grid asset

### Paper B: Three-Mode Grid-Forming Control (Shamseldein, 2025)
**Title:** "From Liability to Asset: A Three-Mode Grid-Forming Control Framework for Centralized Data Center UPS Systems"
**URL:** https://arxiv.org/abs/2512.16497
**Key insight:** Three operating modes for battery

### Paper C: Battery-Assisted Hyperscale AI Data Centers (Lu et al., 2026)
**Title:** "Battery-Assisted Operation of Hyperscale AI Data Centers under Connect-and-Manage Interconnection Practices"
**URL:** https://arxiv.org/abs/2605.14105
**Key insight:** Two-stage stochastic dispatch for AI data centers

### Paper D: Online Feedback Optimization (Mao et al., 2026)
**Title:** "Online Feedback Optimization of Energy Storage to Smooth Data Center Grid Impacts"
**URL:** https://arxiv.org/abs/2603.20564
**Key insight:** Real-time voltage-based battery control

### Paper E: Critical Review of ESS (Mohammadi et al., 2026)
**Title:** "Grid Integration of AI Data Centers: A Critical Review of Energy Storage Solutions"
**URL:** https://arxiv.org/abs/2603.00415
**Key insight:** Four-layer hierarchical ESS taxonomy

### Paper F: Coordinated Fast Frequency Response (Tao & Gadh, 2025)
**Title:** "Coordinated Fast Frequency Response from Electric Vehicles, Data Centers, and Battery Energy Storage Systems"
**URL:** https://arxiv.org/abs/2512.14136
**Key insight:** Multi-resource frequency response coordination

### Paper G: Carbon-Aware Scheduling (Zhang et al., 2026)
**Title:** "Carbon-Aware Compute-Power Scheduling for AI Data Centers with Microgrid Prosumer Operations"
**URL:** https://arxiv.org/abs/2605.03751
**Key insight:** Joint optimization of compute + battery + grid

---

## Signal Classification Methods

### 1. Frequency Regulation Signal Decomposition (RegA vs RegD)

**RegA (Regulation A):**
- Slow, sustained frequency regulation
- Gradual ramping
- Low-frequency grid balancing
- Requires sustained energy delivery

**RegD (Regulation D):**
- Fast, dynamic signal
- Rapid direction changes
- High-frequency power response
- Energy-neutral over short windows

**Classification method:**
- Low-pass filter → RegA-type (sustained energy)
- High-pass filter → RegD-type (fast response)

### 2. Three-Mode Classification (Shamseldein, 2025)

**Mode 1 (Normal):**
- Regulates DC stiff bus
- Manages grid power draw
- Active filtering during normal operation

**Mode 2 (Fault):**
- Current-limited fault-mode P-Q priority
- UPS-BESS buffering
- Rate-limited post-fault "soft return"

**Mode 3 (Grid Support):**
- Droop-based fast frequency response
- Grid-draw modulation
- UPS acts as grid-forming asset

**Signal classification trigger:**
- Grid voltage/frequency measurements
- PCC voltage below thresholds (tested at 0.5 p.u. three-phase dip for 150 ms)

### 3. Our Signal Classification (SAFE/PREPARE/CRITICAL)

Based on the research, we can classify signals as:

**SAFE (Mode 1 - Normal):**
- Grid voltage: 0.95-1.05 p.u.
- Grid frequency: 59.95-60.05 Hz
- Battery SoC: 50-100%
- GPU power: < 80% of capacity

**PREPARE (Mode 2 - Warning):**
- Grid voltage: 0.90-0.95 or 1.05-1.10 p.u.
- Grid frequency: 59.90-59.95 or 60.05-60.10 Hz
- Battery SoC: 20-50%
- GPU power: 80-90% of capacity

**CRITICAL (Mode 3 - Emergency):**
- Grid voltage: < 0.90 or > 1.10 p.u.
- Grid frequency: < 59.90 or > 60.10 Hz
- Battery SoC: < 20%
- GPU power: > 90% of capacity

---

## Battery Dispatch Strategies

### 1. Two-Stage Stochastic Dispatch (Lu et al., 2026)

**Stage 1 - Day-ahead:**
- Scenario-based workload commitment
- BESS capacity allocation across forecasted scenarios
- Grid interconnection limits + computing demand

**Stage 2 - Real-time:**
- Receding-horizon delivery assurance controller
- Battery, thermal, grid-interaction constraints at each time step

**Regime-dependent role transition:**
- When PCC limits binding: BESS provides feasibility-oriented continuity support
- When transmission constraints relax: BESS shifts to economy-driven flexibility

### 2. Online Feedback Optimization (Mao et al., 2026)

**Controller logic:**
1. Responds to real-time voltage measurements at PCC
2. Adjusts active (P) and reactive (Q) power setpoints
3. Minimizes voltage constraint violations
4. Smooths voltage profiles
5. Long-term consistent voltage regulation

### 3. Coordinated Multi-Resource Dispatch (Tao & Gadh, 2025)

**Resource classification by response speed:**
- BESS: Sub-second response
- UPS workload modulation: Seconds
- EV charging adjustment: Minutes

**Dynamic allocation:**
- Upper-level coordinator performs dynamic allocation
- Lower-level controllers implement power response
- Result: Frequency nadir improved by up to 0.2 Hz

### 4. MILP Joint Optimization (Zhang et al., 2026)

**Joint optimization of:**
- Scheduling rigid training jobs
- Routing elastic inference workloads across sites
- Dispatching local generation and battery storage
- Managing bidirectional grid interaction

**Constraints:**
- Latency
- Continuity
- Power-balance
- Carbon-budget

### 5. Four-Layer Hierarchical ESS Dispatch (Mohammadi et al., 2026)

**Layer 1 - Chip-level buffering:**
- Fastest, most granular (microsecond response)

**Layer 2 - Rack/server-level ESSs:**
- Intermediate scale (millisecond response)

**Layer 3 - Facility-level UPS systems:**
- Building-scale (cycle-to-cycle response)

**Layer 4 - Grid-scale BESS:**
- Utility-scale (seconds to minutes)

**Key finding:** AI data center loads have sub-second variability. Coordinated deployment across all four layers is necessary.

---

## Core Optimization Formulas

### 1. Peak Shaving Objective

```
min  max(P_grid(t))
where P_grid(t) = P_load(t) - P_BESS(t)
subject to:
  SoC_min <= SoC(t) <= SoC_max
  P_min <= P_BESS(t) <= P_max
  SoC(t+1) = SoC(t) + (η_c * P_c(t) - P_d(t)/η_d) * dt / E_cap
```

### 2. Battery State-of-Charge Dynamics

```
SoC(t+1) = SoC(t) + (η_c * P_c(t) - P_d(t)/η_d) * Δt / E_cap
```

Where:
- η_c = charging efficiency
- η_d = discharging efficiency
- P_c(t) = charging power
- P_d(t) = discharging power
- E_cap = total energy capacity
- Δt = time step duration

### 3. SoC Estimation (Coulomb Counting)

```
SoC(t) = SoC(t_0) - (1/Q_n) * ∫I(τ)dτ
```

Where Q_n is nominal capacity and I(τ) is current over time.

### 4. Battery Degradation Model (Semi-empirical)

```
Q_loss = A * exp(-E_a / (R*T)) * (Ah)^z
```

Where:
- A = pre-exponential factor
- E_a = activation energy
- R = gas constant
- T = temperature
- Ah = amp-hours throughput
- z = aging exponent

### 5. Multi-Objective Dispatch Optimization

```
min  Σ(C_deg * d * |P(t)|) + Σ(C_arb * P_arb(t))
```

Where:
- C_deg = degradation cost per cycle
- d = depth of discharge factor
- C_arb = arbitrage cost/revenue
- P_arb(t) = arbitrage power at time t

### 6. Model Predictive Control (MPC)

At each time step k, solve over rolling horizon N:

```
min  Σ_{t=k}^{k+N} [ C_elec(t) * P_grid(t) + C_deg * |P_BESS(t)| ]
subject to:
  SoC(t+1) = SoC(t) + f(P_BESS(t))
  SoC_min <= SoC(t) <= SoC_max
  P_BESS,min <= P_BESS(t) <= P_BESS,max
  P_grid(t) = P_load(t) - P_BESS(t)
  P_grid(t) <= P_peak_max  (peak shaving constraint)
```

### 7. Droop-Based Grid-Forming Control

```
P = P_0 + k_p * (f_0 - f)
Q = Q_0 + k_q * (V_0 - V)
```

Where k_p, k_q are droop gains, f_0, V_0 are nominal frequency and voltage.

---

## Tesla Megapack Specifications

| Spec | Value |
|------|-------|
| Energy capacity | ~3.9 MWh per unit |
| Power output | ~1.9 MW per unit |
| Dimensions | ~23 ft × 5.5 ft × 9 ft |
| Weight | ~80,000 lbs (36,000 kg) |
| Inverter | Integrated bi-directional |
| Cooling | Liquid cooling |
| Warranty | 20 years |

### Tesla Autobidder Platform

- Real-time power trading
- Demand forecasting
- Autonomous charge/discharge scheduling
- Revenue stacking: energy arbitrage, ancillary services, capacity payments, frequency regulation
- Manages >1.2 GWh of storage (as of March 2021)

**Note:** Algorithm details are proprietary. No public code.

---

## Implementation Plan for ENERGIVANU

### Step 1: Signal Classification Features

```python
def classify_signal(power, voltage, frequency, soc, capacity):
    """
    Classify battery signal as SAFE/PREPARE/CRITICAL

    Args:
        power: Current GPU power (MW)
        voltage: Grid voltage (p.u.)
        frequency: Grid frequency (Hz)
        soc: Battery state of charge (0-1)
        capacity: Total GPU capacity (MW)

    Returns:
        signal: 0=SAFE, 1=PREPARE, 2=CRITICAL
    """
    power_ratio = power / capacity

    # CRITICAL conditions
    if (voltage < 0.90 or voltage > 1.10 or
        frequency < 59.90 or frequency > 60.10 or
        soc < 0.20 or power_ratio > 0.90):
        return 2

    # PREPARE conditions
    if (voltage < 0.95 or voltage > 1.05 or
        frequency < 59.95 or frequency > 60.05 or
        soc < 0.50 or power_ratio > 0.80):
        return 1

    # SAFE
    return 0
```

### Step 2: Battery Dispatch Optimization

```python
def optimize_dispatch(current_soc, forecast_power, grid_price,
                      degradation_cost, horizon=60):
    """
    Optimize battery dispatch using MPC

    Args:
        current_soc: Current state of charge (0-1)
        forecast_power: Predicted GPU power for next horizon (MW)
        grid_price: Electricity price signal ($/MWh)
        degradation_cost: Battery degradation cost ($/MWh)
        horizon: Optimization horizon (minutes)

    Returns:
        dispatch: Battery power setpoint for each timestep (MW)
    """
    # Use CVXPY or Pyomo for optimization
    import cvxpy as cp

    T = horizon
    P_batt = cp.Variable(T)  # Battery power (positive=discharge)
    SoC = cp.Variable(T+1)   # State of charge

    # Constraints
    constraints = [
        SoC[0] == current_soc,
        SoC >= 0.1,  # Min SoC
        SoC <= 0.9,  # Max SoC
        P_batt >= -1.9,  # Max charge rate (MW)
        P_batt <= 1.9,   # Max discharge rate (MW)
    ]

    # SoC dynamics
    for t in range(T):
        constraints.append(
            SoC[t+1] == SoC[t] - P_batt[t] * (1/60) / 3.9  # 3.9 MWh capacity
        )

    # Objective: minimize grid cost + degradation
    grid_power = forecast_power - P_batt
    objective = cp.Minimize(
        cp.sum(cp.multiply(grid_price, grid_power)) +
        degradation_cost * cp.sum(cp.abs(P_batt))
    )

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return P_batt.value
```

### Step 3: Direction Prediction

```python
def predict_direction(power_history, window=12):
    """
    Predict power direction (UP/DOWN)

    Args:
        power_history: Past power values (MW)
        window: Lookback window (1 minute = 12 * 5sec)

    Returns:
        direction: 0=DOWN, 1=UP
    """
    # Use rolling momentum
    if len(power_history) < window:
        return 0

    recent_avg = np.mean(power_history[-window:])
    older_avg = np.mean(power_history[-2*window:-window])

    return 1 if recent_avg > older_avg else 0
```

---

## Key Takeaways

1. **Signal classification** operates at multiple timescales (sub-second to minutes)
2. **Battery dispatch** must balance IT continuity, grid revenue, and degradation
3. **Four-layer hierarchy** is essential for AI data center loads
4. **MPC optimization** is the standard approach for battery dispatch
5. **Tesla Megapack/Autobidder** details are proprietary
6. **Our SAFE/PREPARE/CRITICAL** maps to three-mode classification from research

---

## Sources

- https://arxiv.org/abs/2512.16497 (Shamseldein 2025)
- https://arxiv.org/abs/2605.14105 (Lu et al. 2026)
- https://arxiv.org/abs/2603.20564 (Mao et al. 2026)
- https://arxiv.org/abs/2603.00415 (Mohammadi et al. 2026)
- https://arxiv.org/abs/2512.14136 (Tao & Gadh 2025)
- https://arxiv.org/abs/2605.03751 (Zhang et al. 2026)
- https://www.sciencedirect.com/science/article/pii/S0142061526000803 (Azizi et al. 2026)
