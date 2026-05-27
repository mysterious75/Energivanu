# ENERGIVANU — Combined Solution Architecture

## The Big Picture

No one has solved our EXACT problem. But we have ALL the pieces. Our job: combine them.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENERGIVANU SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   LAYER 1   │    │   LAYER 2   │    │   LAYER 3   │         │
│  │   Power     │    │   Signal    │    │  Direction  │         │
│  │ Prediction  │    │Classification│   │ Prediction  │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ PI-DLinear  │    │  Rule-Based │    │  Feature    │         │
│  │ + Thermal   │    │  + ML       │    │ Engineering │         │
│  │   RC Net    │    │  Classifier │    │  Momentum   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                    ┌─────────────┐                              │
│                    │   LAYER 4   │                              │
│                    │  Battery    │                              │
│                    │  Dispatch   │                              │
│                    │  (MPC)      │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │  Tesla      │                              │
│                    │  Megapack   │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Piece 1: Power Prediction (PI-DLinear)

**Source:** arXiv:2605.04074

**What it does:** Predicts GPU power 10 minutes ahead

**How it works:**
1. DLinear decomposes time series into trend + seasonal
2. Thermal RC network adds physics constraints
3. Self-adaptive weighting balances data vs physics loss

**Our implementation:**
```python
class PIDLinear(nn.Module):
    def __init__(self, lookback=60, horizon=60):
        super().__init__()
        # DLinear components
        self.trend_linear = nn.Linear(lookback, horizon)
        self.seasonal_linear = nn.Linear(lookback, horizon)

        # Thermal RC network (physics)
        self.thermal = ThermalRCNetwork()

        # Self-adaptive weighting
        self.weighting = SelfAdaptiveWeighting()

    def forward(self, x, T_g, T_m, P, T_a=27.0):
        # Decompose
        trend = x.mean(dim=-1, keepdim=True)
        seasonal = x - trend

        # Linear projections
        h_trend = self.trend_linear(trend)
        h_seasonal = self.seasonal_linear(seasonal)

        # Physics-informed prediction
        dT_g, dT_m = self.thermal(T_g, T_m, P, T_a)

        # Combine
        pred = h_trend + h_seasonal

        return pred, dT_g, dT_m
```

**Expected result:** MAE < 3 MW (proven in paper)

---

## Piece 2: Signal Classification (Rule-Based + ML)

**Source:** Shamseldein 2025, Lu et al. 2026

**What it does:** Classifies battery signal as SAFE/PREPARE/CRITICAL

**How it works:**
1. Rule-based classification from grid conditions
2. ML refinement from historical patterns
3. Multi-timescale classification

**Our implementation:**
```python
class SignalClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # SAFE/PREPARE/CRITICAL

    def forward(self, x):
        # Features: power_ratio, voltage, frequency, soc, rate_of_change, etc.
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)  # Logits for 3 classes
```

**Signal features:**
| Feature | Description | Source |
|---------|-------------|--------|
| power_ratio | Current power / capacity | Our data |
| rate_of_change | Power derivative | Our data |
| volatility | Power variance (5min window) | Our data |
| soc | Battery state of charge | Tesla API |
| voltage | Grid voltage | Grid API |
| frequency | Grid frequency | Grid API |

**Expected result:** SigAcc > 95%

---

## Piece 3: Direction Prediction (Feature Engineering)

**Source:** Our research + market solutions

**What it does:** Predicts if power will go UP or DOWN

**How it works:**
1. Feature engineering: momentum, ROC, volatility
2. Simple classifier (not regression)
3. Separate from power prediction

**Our implementation:**
```python
class DirectionPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # UP/DOWN

    def forward(self, x):
        # Features: momentum, ROC, volatility, etc.
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)  # Logits for UP/DOWN
```

**Direction features:**
| Feature | Description |
|---------|-------------|
| momentum_1m | 1-minute rolling average change |
| momentum_5m | 5-minute rolling average change |
| roc_1m | 1-minute rate of change |
| roc_5m | 5-minute rate of change |
| volatility_5m | 5-minute standard deviation |
| trend_strength | Linear regression slope |

**Expected result:** DirAcc > 55%

---

## Piece 4: Battery Dispatch (MPC Optimization)

**Source:** Lu et al. 2026, Mao et al. 2026

**What it does:** Optimizes Tesla Megapack charge/discharge

**How it works:**
1. Model Predictive Control (MPC)
2. Multi-objective: grid cost + degradation + continuity
3. Rolling horizon optimization

**Our implementation:**
```python
class BatteryDispatchMPC:
    def __init__(self, capacity_mwh=3.9, max_power_mw=1.9,
                 degradation_cost=10.0):
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.deg_cost = degradation_cost

    def optimize(self, soc, forecast_power, grid_price, horizon=60):
        """
        Optimize battery dispatch for next horizon

        Args:
            soc: Current state of charge (0-1)
            forecast_power: Predicted GPU power (MW) for horizon
            grid_price: Electricity price ($/MWh) for horizon
            horizon: Optimization window (minutes)

        Returns:
            dispatch: Battery power setpoint (MW) for each minute
        """
        import cvxpy as cp

        T = horizon
        P_batt = cp.Variable(T)
        SoC = cp.Variable(T+1)

        constraints = [
            SoC[0] == soc,
            SoC >= 0.1,
            SoC <= 0.9,
            P_batt >= -self.max_power,
            P_batt <= self.max_power,
        ]

        for t in range(T):
            constraints.append(
                SoC[t+1] == SoC[t] - P_batt[t] * (1/60) / self.capacity
            )

        grid_power = forecast_power - P_batt
        objective = cp.Minimize(
            cp.sum(cp.multiply(grid_price, grid_power)) +
            self.deg_cost * cp.sum(cp.abs(P_batt))
        )

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return P_batt.value
```

**Expected result:** Optimal battery usage, reduced grid costs

---

## Combined System: How Pieces Fit Together

### Step 1: Data Collection (Every 5 seconds)
```
GPU Power, Temperature, Utilization, Memory → Feature Store
```

### Step 2: Power Prediction (Every minute)
```
PI-DLinear → Power forecast for next 10 minutes
```

### Step 3: Signal Classification (Every minute)
```
SignalClassifier → SAFE/PREPARE/CRITICAL
```

### Step 4: Direction Prediction (Every minute)
```
DirectionPredictor → UP/DOWN
```

### Step 5: Battery Dispatch (Every minute)
```
MPC Optimizer → Battery charge/discharge setpoint
```

### Step 6: Execute (Every 5 seconds)
```
Tesla Megapack → Execute setpoint
```

---

## Architecture Options

### Option A: Single Model (Current Approach)
```
Input → TSMixer/Transformer → [Power, Signal, Direction]
```
**Pros:** Simple, end-to-end
**Cons:** Gradient starvation, hard to optimize

### Option B: Multi-Head with Shared Backbone
```
Input → Shared Encoder → [Power Head, Signal Head, Direction Head]
```
**Pros:** Shared features, specialized heads
**Cons:** Still gradient starvation risk

### Option C: Cascaded Models (Recommended)
```
Input → PI-DLinear → Power forecast
                   → Signal classifier → Signal
                   → Direction predictor → Direction
```
**Pros:** No gradient starvation, specialized models, physics constraints
**Cons:** More complex pipeline

### Option D: Ensemble
```
Input → [PI-DLinear, TSMixer, PatchTST] → Average → Output
```
**Pros:** Best accuracy, robust
**Cons:** Slower inference, more memory

---

## Recommended Implementation Order

| Phase | Task | Priority | Time |
|-------|------|----------|------|
| 1 | PI-DLinear implementation | P0 | 2 days |
| 1 | Direction features engineering | P0 | 1 day |
| 1 | Signal classifier training | P0 | 1 day |
| 2 | Battery dispatch MPC | P1 | 2 days |
| 2 | Integration testing | P1 | 1 day |
| 3 | Ensemble with TSMixer | P2 | 1 day |
| 3 | Foundation model fine-tuning | P2 | 2 days |
| 4 | Production deployment | P3 | 3 days |

---

## Success Criteria

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| MAE | 3.82 MW | < 3 MW | PI-DLinear |
| SigAcc | 93.6% | > 95% | Rule-based + ML |
| DirAcc | 56.0% | > 55% | Feature engineering |
| Inference | Unknown | < 100ms | Optimized pipeline |
| Battery | N/A | Optimal | MPC dispatch |

---

## Key Formulas Summary

### 1. Thermal RC ODEs
```
C_g * dT_g/dt = α*P - (T_g - T_a)/R_ga - (T_g - T_m)/R_gm
C_m * dT_m/dt = (1-α)*P - (T_m - T_a)/R_ma + (T_g - T_m)/R_gm
```

### 2. Physics Loss
```
L = λ_data * L_MSE + λ_physics * L_RC + λ_throttle * L_throttle
```

### 3. Signal Classification
```
SAFE:     voltage ∈ [0.95, 1.05], freq ∈ [59.95, 60.05], SoC > 50%
PREPARE:  voltage ∈ [0.90, 0.95] ∪ [1.05, 1.10], SoC ∈ [20%, 50%]
CRITICAL: voltage < 0.90 ∪ > 1.10, freq < 59.90 ∪ > 60.10, SoC < 20%
```

### 4. Direction Prediction
```
UP:   momentum_1m > 0 AND roc_1m > 0
DOWN: momentum_1m < 0 OR roc_1m < 0
```

### 5. Battery SoC Dynamics
```
SoC(t+1) = SoC(t) + (η_c * P_c(t) - P_d(t)/η_d) * Δt / E_cap
```

### 6. MPC Objective
```
min Σ[C_elec(t) * P_grid(t) + C_deg * |P_BESS(t)|]
```

---

## Sources

- PI-DLinear: https://arxiv.org/abs/2605.04074
- Signal Classification: https://arxiv.org/abs/2512.16497
- Battery Dispatch: https://arxiv.org/abs/2605.14105
- MPC Optimization: https://arxiv.org/abs/2603.20564
- Four-Layer ESS: https://arxiv.org/abs/2603.00415
- Coordinated FFR: https://arxiv.org/abs/2512.14136
- Carbon-Aware: https://arxiv.org/abs/2605.03751
- Grid-Forming UPS: https://www.sciencedirect.com/science/article/pii/S0142061526000803
