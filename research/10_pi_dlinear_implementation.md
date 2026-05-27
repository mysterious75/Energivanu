# ENERGIVANU — PI-DLinear Implementation Guide

## Paper Details

**Title:** "A Physics-Aware Framework for Short-Term GPU Power Forecasting of AI Data Centers"
**arXiv:** https://arxiv.org/abs/2605.04074
**Authors:** Mohammad AlShaikh Saleh, Sanjay Chawla, Sertac Bayhan, Haitham Abu-Rub, Ali Ghrayeb
**Year:** 2026
**License:** CC BY 4.0

---

## Core Concept

PI-DLinear = Physics-Informed DLinear
- DLinear: Decomposition-based Linear model (trend + seasonal)
- Physics: Thermal RC network constraints
- Result: 0.78-39% better MSE than transformers

---

## Feature Vector (5 co-variates)

```
x_t = [u_t^(g), u_t^(m), T_t^(g), T_t^(m), P_t]
```

| Feature | Description |
|---------|-------------|
| u_t^(g) | GPU utilization |
| u_t^(m) | Memory utilization |
| T_t^(g) | GPU temperature |
| T_t^(m) | Memory temperature |
| P_t | Electrical power |

---

## Thermal RC Network ODEs (Two-Node Model)

### GPU Node:
```
C_g * dT_g/dt = alpha*P - (T_g - T_a)/R_ga - (T_g - T_m)/R_gm
```

### Memory Node:
```
C_m * dT_m/dt = (1-alpha)*P - (T_m - T_a)/R_ma + (T_g - T_m)/R_gm
```

### Variables:
| Symbol | Description | Value (MIT Supercloud) |
|--------|-------------|------------------------|
| C_g | GPU thermal capacitance | 5.408 × 10^6 J/K |
| C_m | Memory thermal capacitance | 5.481 × 10^6 J/K |
| R_ga | GPU-to-ambient resistance | 2.037 × 10^-3 K/W |
| R_ma | Memory-to-ambient resistance | 2.055 × 10^-3 K/W |
| R_gm | GPU-memory coupling resistance | 6.064 × 10^-4 K/W |
| alpha | Power split parameter | 0.5085 |
| T_a | Ambient temperature | 27°C (constant) |

---

## Power Rate Constraint

Derived by solving GPU ODE for P and differentiating:

```
dP/dt = (1/alpha) * [C_g * d²T_g/dt² + (1/R_ga)*dT_g/dt + (1/R_gm)*(dT_g/dt - dT_m/dt)]
```

---

## DLinear Decomposition

```
H_s = W_s * X_s  (seasonal)
H_t = W_t * X_t  (trend)
y_hat = H_s + H_t
```

Where:
- W_s, W_t: Single-layer linear networks
- X_s: Seasonal component (via moving average)
- X_t: Trend component (via moving average)

---

## Physics-Informed Loss Function

### Total Loss:
```
L = lambda_u * L_Data + lambda_r * L_r + lambda_theta * L_throttle
```

### Data Loss (MSE):
```
L_Data = (1/N_u) * Σ|P̂(x_u^i, t_u^i) - P_i|²
```

### Physics Residual Loss:
```
L_r = (1/N_r) * Σ|P(x_r^i, t_r^i)|²
```

### Power Throttling Constraint:
```
L_throttle = L_high + L_stress

L_high = (1/(H-1)) * Σ_{t: U_t > θ_U} max(0, dP̂_t)²

L_stress = (1/(H-1)) * Σ_{t: U_t > θ_U, T_t^g > θ_T} max(0, dP̂_t)²
```

Where:
- dP̂_t = P̂_{t+1} - P̂_t
- U_t = α * u_t^(g) + (1-α) * u_t^(m)
- θ_U ≈ 90% (utilization threshold)
- θ_T = 95th percentile temperature
- Throttle events = "sudden power drops exceeding 15%"

---

## Self-Adaptive Weighting

Weights updated via gradient ascent in log-space:

```
η_u = log(λ_u)
η_r = log(λ_r)
η_θ = log(λ_θ)

η ← η + γ * ∇_η L
```

With η clipped to [λ_min, λ_max].

---

## Results

### Dataset: MIT Supercloud Dataset
- NVIDIA Volta V100 GPUs
- ~330,500 timesteps at 1-minute granularity
- ~238 days (Feb-Oct 2021)
- Peak power ~45 kW across 448 GPUs

### Performance (averaged across prediction horizons, T=240 min look-back):

| Model | MAE | MSE | MAPE | RMSE |
|-------|-----|-----|------|------|
| PI-DLinear | 0.1420 | 0.1546 | 1.0315 | 0.3895 |
| DLinear | 0.1420 | 0.1556 | 1.0403 | 0.3907 |
| FiLM | 0.1432 | 0.1571 | 1.0708 | 0.3925 |
| TiDE | 0.1422 | 0.1561 | 1.0628 | 0.3912 |

### Improvement over SOTA baselines:
- MSE: 0.782%-39.08%
- MAE: 0.993%-51.82%
- RMSE: 0.370%-22.28%

### Power Throttling Detection:
- Detection rate improved by 6.88% on average (best: 19.75%)
- Near-perfect detection: 99.12% at L=480, H=80

### Computational Efficiency:

| Model | #Params | Time (s/epoch) | Memory (MB) |
|-------|---------|----------------|-------------|
| DLinear | 96,160 | 10.43 | 0.376 |
| PI-DLinear | 96,160 | 20.27 | 0.376 |
| FiLM | 12,923,662 | 271.38 | 49.30 |

**Key insight:** PI-DLinear preserves DLinear's parameter count. Physics loss only used in training, NOT inference.

---

## Implementation Plan for ENERGIVANU

### Step 1: Get Features
We need: GPU utilization, memory utilization, GPU temp, memory temp, power
- Source: DCGM (NVIDIA Data Center GPU Manager)
- Alternative: Synthetic data with these features

### Step 2: Implement Thermal RC Network
```python
class ThermalRCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable RC parameters
        self.C_g = nn.Parameter(torch.tensor(5.408e6))
        self.C_m = nn.Parameter(torch.tensor(5.481e6))
        self.R_ga = nn.Parameter(torch.tensor(2.037e-3))
        self.R_ma = nn.Parameter(torch.tensor(2.055e-3))
        self.R_gm = nn.Parameter(torch.tensor(6.064e-4))
        self.alpha = nn.Parameter(torch.tensor(0.5085))

    def forward(self, T_g, T_m, P, T_a=27.0):
        # GPU node ODE
        dT_g = (self.alpha * P - (T_g - T_a)/self.R_ga -
                (T_g - T_m)/self.R_gm) / self.C_g

        # Memory node ODE
        dT_m = ((1-self.alpha) * P - (T_m - T_a)/self.R_ma +
                (T_g - T_m)/self.R_gm) / self.C_m

        return dT_g, dT_m
```

### Step 3: Implement Physics Loss
```python
def physics_loss(pred_power, dT_g, dT_m, dt=60.0):
    # Power rate constraint
    dP_dt = (pred_power[:, 1:] - pred_power[:, :-1]) / dt

    # Physics residual (ODE should be satisfied)
    residual = dP_dt - (1/alpha) * (
        C_g * d²T_g/dt² +
        (1/R_ga) * dT_g/dt +
        (1/R_gm) * (dT_g/dt - dT_m/dt)
    )

    return residual.pow(2).mean()
```

### Step 4: Implement Throttling Loss
```python
def throttling_loss(pred_power, utilization, temperature,
                    theta_U=0.9, theta_T_percentile=95):
    dP = pred_power[:, 1:] - pred_power[:, :-1]

    # High utilization constraint
    high_util = utilization[:, 1:] > theta_U
    L_high = (F.relu(dP) * high_util.float()).pow(2).mean()

    # Stress constraint (high util + high temp)
    theta_T = torch.quantile(temperature, theta_T_percentile/100)
    stress = high_util & (temperature[:, 1:] > theta_T)
    L_stress = (F.relu(dP) * stress.float()).pow(2).mean()

    return L_high + L_stress
```

### Step 5: Self-Adaptive Weighting
```python
class SelfAdaptiveWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lambda_data = nn.Parameter(torch.zeros(1))
        self.log_lambda_physics = nn.Parameter(torch.zeros(1))
        self.log_lambda_throttle = nn.Parameter(torch.zeros(1))

    def forward(self, L_data, L_physics, L_throttle):
        lambda_data = torch.exp(self.log_lambda_data)
        lambda_physics = torch.exp(self.log_lambda_physics)
        lambda_throttle = torch.exp(self.log_lambda_throttle)

        total = (lambda_data * L_data +
                 lambda_physics * L_physics +
                 lambda_throttle * L_throttle +
                 self.log_lambda_data +
                 self.log_lambda_physics +
                 self.log_lambda_throttle)

        return total
```

---

## Key Differences from Our Current Approach

| Aspect | Current | PI-DLinear |
|--------|---------|------------|
| Features | 34 features | 5 features (physics-relevant) |
| Architecture | TSMixer/Transformer | DLinear (simple linear) |
| Loss | MSE + CE + Direction | MSE + Physics + Throttling |
| Parameters | 360K | 96K |
| Training time | 50s/epoch | 20s/epoch |
| Physics constraints | None | Thermal RC network |

---

## Advantages of PI-DLinear

1. **Fewer parameters** (96K vs 360K) = less overfitting
2. **Physics constraints** = better generalization
3. **Faster training** (20s vs 50s per epoch)
4. **Interpretable** = can understand physics
5. **Proven results** = 0.78-39% better MSE

---

## Next Steps

1. Add GPU temp, memory temp, utilization features to our data
2. Implement ThermalRCNetwork class
3. Implement physics_loss and throttling_loss
4. Implement SelfAdaptiveWeighting
5. Train PI-DLinear on our data
6. Compare with current TSMixer results

---

## Sources

- https://arxiv.org/abs/2605.04074
- https://arxiv.org/pdf/2605.04074
- MIT Supercloud Dataset: https://supercloud.mit.edu
