# CVXPY MPC Controller

## Overview
Model Predictive Control using CVXPY quadratic programming solver for optimal battery dispatch.

## Problem Formulation
```
minimize:   Q * ||P_grid - P_target||² + R * ||u||² + S * (SOC[N] - 0.5)²
subject to:
  SOC[k+1] = SOC[k] - u[k] * dt / capacity * efficiency
  SOC_min ≤ SOC[k] ≤ SOC_max
  -P_battery_max ≤ u[k] ≤ P_battery_max
  P_grid[k] = P_load[k] - u[k] ≥ 0
```

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| horizon | 12 steps | Prediction horizon |
| dt | 5 seconds | Time step |
| SOC min | 5% | Minimum state of charge |
| SOC max | 95% | Maximum state of charge |
| efficiency | 92% | Round-trip efficiency |
| battery power | 319.2 MW | Max charge/discharge rate |
| battery capacity | 655.2 MWh | Total capacity (Tesla Megapack scale) |
| grid target | 200 MW | Target grid power |
| Q | 100.0 | Tracking weight |
| R | 0.01 | Control effort weight |
| S | 0.1 | Terminal SOC weight |
| solver | OSQP | Quadratic programming solver |

## Results
- **Peak reduction:** 6.36 MW (on 100 test samples)
- **Battery SOC maintained:** 5-95%
- **Grid power smoothed** within target range

## Integration
1. Model predicts facility power for next 10 steps
2. MPC extends prediction to 12 steps (repeat last)
3. MPC optimizes battery charge/discharge schedule
4. Battery command applied to BESS hardware
5. Repeat every 5 seconds
