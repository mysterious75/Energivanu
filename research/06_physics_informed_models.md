# Physics-Informed Machine Learning Models for Energy/Power Forecasting

**Project:** ENERGIVANU
**Document:** 06 - Physics-Informed Models Research
**Date:** 2026-05-27

---

## Table of Contents

1. [Physics-Informed Neural Networks (PINNs)](#1-physics-informed-neural-networks-pinns)
2. [Thermal Modeling for GPUs](#2-thermal-modeling-for-gpus)
3. [Power Modeling](#3-power-modeling)
4. [Physics-Informed Loss Functions](#4-physics-informed-loss-functions)
5. [Hybrid Models](#5-hybrid-models)
6. [Battery Modeling](#6-battery-modeling)
7. [Grid Stability Constraints](#7-grid-stability-constraints)
8. [Implementation Architecture](#8-implementation-architecture)
9. [References](#9-references)

---

## 1. Physics-Informed Neural Networks (PINNs)

### 1.1 How PINNs Work

Physics-Informed Neural Networks (PINNs) are deep neural networks trained to solve supervised learning tasks while respecting physical laws described by general nonlinear partial differential equations (PDEs). They serve as data-efficient universal function approximators that naturally encode underlying physical laws as prior information.

**Core Mechanism:**

A PINN approximates the solution u(t,x) using a deep neural network, then constructs a second network f(t,x) derived from the governing PDE using automatic differentiation. For a general PDE of the form:

```
u_t + N[u] = 0
```

The physics-informed network is defined as:

```
f := u_t + N[u]
```

where the partial derivatives are computed via automatic differentiation through the computational graph (e.g., PyTorch's `autograd` or TensorFlow's `GradientTape`).

**Two Problem Classes:**

| Class | Description | Use Case |
|-------|-------------|----------|
| **Data-Driven Solution** | Given known PDE parameters, infer the solution field u(t,x) | Forward simulation, forecasting |
| **Data-Driven Discovery** | Given noisy measurements, simultaneously compute u(t,x) and learn unknown PDE parameters lambda | System identification, parameter estimation |

### 1.2 Incorporating Physical Constraints

Physical constraints are incorporated through multiple mechanisms:

**1. Direct PDE Enforcement:**
The PDE residual is evaluated at randomly sampled "collocation points" in the spatio-temporal domain. These points are typically generated via Latin Hypercube Sampling for uniform coverage.

**2. Automatic Differentiation:**
Derivatives such as u_t, u_x, u_xx are computed exactly through the neural network's computational graph. This is superior to numerical or symbolic differentiation for smooth functions.

**3. Boundary and Initial Conditions:**
Hard or soft enforcement of boundary conditions through dedicated loss terms or architectural constraints (e.g., using the Theory of Functional Connections to analytically satisfy constraints).

**4. Conservation Laws:**
Divergence-free conditions, energy conservation, and mass conservation can be encoded as PDE residuals or architectural inductive biases (e.g., stream function formulation for incompressible flow).

### 1.3 Loss Function Design

The total loss function combines data fidelity and physics enforcement:

```
L_total = L_data + L_physics + L_boundary + L_initial
```

**Component Breakdown:**

```python
# L_data: Measured data fidelity
L_data = (1/N_d) * sum_i ||u(t_i, x_i) - u_observed_i||^2

# L_physics: PDE residual at collocation points
L_physics = (1/N_c) * sum_j ||f(t_j, x_j)||^2
# where f = du/dt + N[u; lambda] is the PDE residual

# L_boundary: Boundary condition enforcement
L_boundary = (1/N_b) * sum_k ||u(t_k, x_boundary_k) - g_k||^2

# L_initial: Initial condition enforcement
L_initial = (1/N_0) * sum_m ||u(t_0, x_m) - u_0(x_m)||^2
```

**Weighting Strategies:**

The balance between loss terms is critical. Common approaches include:

- **Fixed weights:** L_total = w_data * L_data + w_physics * L_physics + w_bc * L_bc
- **Adaptive weights:** Learned during training (e.g., self-adaptive PINNs)
- **Curriculum training:** Start with data loss, gradually increase physics loss weight
- **Gradient balancing:** Normalize gradient magnitudes across loss terms

### 1.4 Advantages Over Pure Data-Driven Models

| Advantage | Description |
|-----------|-------------|
| **Data efficiency** | Requires significantly less training data by encoding physical priors |
| **Generalization** | Physics constraints limit the admissible solution space, preventing unphysical predictions |
| **Consistency** | Predictions respect conservation laws and other physical principles |
| **Interpretability** | Learned parameters have physical meaning |
| **Mesh-free** | No spatial/temporal discretization required (unlike FEM/FDM) |
| **Multi-fidelity** | Can integrate datasets of varying quality and quantity |
| **Inverse problems** | Can simultaneously learn unknown model parameters from data |
| **Uncertainty quantification** | Bayesian variants (B-PINNs) provide uncertainty estimates |

### 1.5 PINN Implementation in PyTorch

```python
import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for energy system modeling.
    """
    def __init__(self, layers, activation='tanh'):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                if activation == 'tanh':
                    self.net.add_module(f"tanh_{i}", nn.Tanh())
                elif activation == 'swish':
                    self.net.add_module(f"swish_{i}", nn.SiLU())

        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def compute_physics_residual(model, t, x, physics_fn):
    """
    Compute PDE residual using automatic differentiation.

    Args:
        model: PINN model
        t: time tensor (requires_grad=True)
        x: spatial/state tensor (requires_grad=True)
        physics_fn: function that computes the PDE residual from u, u_t, u_x, etc.
    """
    input_tensor = torch.cat([t, x], dim=1)
    u = model(input_tensor)

    # First derivatives
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]

    # Second derivative (if needed)
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        retain_graph=True, create_graph=True
    )[0]

    # Compute physics residual
    residual = physics_fn(u, u_t, u_x, u_xx)
    return residual


def total_loss(model, t_data, x_data, u_data, t_colloc, x_colloc,
               physics_fn, w_data=1.0, w_physics=1.0):
    """
    Compute total loss = data loss + physics loss.
    """
    # Data loss
    input_data = torch.cat([t_data, x_data], dim=1)
    u_pred = model(input_data)
    loss_data = torch.mean((u_pred - u_data) ** 2)

    # Physics loss
    residual = compute_physics_residual(model, t_colloc, x_colloc, physics_fn)
    loss_physics = torch.mean(residual ** 2)

    return w_data * loss_data + w_physics * loss_physics
```

### 1.6 Domain Decomposition Variants

For large-scale problems, domain decomposition approaches enable parallelization:

- **cPINNs:** Spatial domain decomposition for conservation laws
- **XPINNs:** Generalized space-time domain decomposition supporting arbitrary PDEs
- **DPINNs/DPIELMs:** Distributed approaches using space-time domain discretization

---

## 2. Thermal Modeling for GPUs

### 2.1 Newton's Law of Cooling

Newton's law states that the rate of heat loss of a body is directly proportional to the difference in temperatures between the body and its environment:

```
q = h * (T(t) - T_env) = h * DeltaT(t)
```

where:
- `q` = heat flux (W/m^2)
- `h` = heat transfer coefficient (W/(m^2*K))
- `T(t)` = body temperature at time t
- `T_env` = ambient/environment temperature

**Differential Equation Form:**

For a body with thermal capacitance C and thermal resistance R:

```
dT/dt = (P_generated - (T - T_ambient) / R) / C
```

where:
- `P_generated` = heat generated (W) -- for GPUs, this equals power consumption
- `R` = thermal resistance (K/W)
- `C` = thermal capacitance (J/K)

**Solution (exponential approach to steady state):**

```
T(t) = T_ambient + R * P_generated * (1 - exp(-t / (R*C)))
```

where `tau = R * C` is the thermal time constant.

### 2.2 Thermal RC Network Model

The thermal behavior of a GPU can be modeled using an electrical analogy:

| Thermal Quantity | Electrical Analog | Units |
|-----------------|-------------------|-------|
| Temperature (T) | Voltage (V) | K or C |
| Heat flow (Q) | Current (I) | W |
| Thermal resistance (R_th) | Resistance (R) | K/W |
| Thermal capacitance (C_th) | Capacitance (C) | J/K |

**Multi-Node Thermal RC Network for GPU:**

A GPU thermal stack typically consists of multiple layers, each modeled as an RC node:

```
Node 1: GPU Die
  - C_die = m_die * c_silicon  (thermal capacitance)
  - R_TIM = thermal resistance of thermal interface material

Node 2: Heat Spreader
  - C_spreader = m_spreader * c_copper
  - R_spreader = thermal resistance through spreader

Node 3: Heatsink Base
  - C_heatsink = m_heatsink * c_aluminum
  - R_convective = 1 / (h_convection * A_surface)

Node 4: Ambient Air
  - T_ambient (boundary condition)
```

**Governing Equations (State-Space Form):**

```
C_1 * dT_1/dt = P_gpu - (T_1 - T_2) / R_12
C_2 * dT_2/dt = (T_1 - T_2) / R_12 - (T_2 - T_3) / R_23
C_3 * dT_3/dt = (T_2 - T_3) / R_23 - (T_3 - T_ambient) / R_3a
```

**Simplified Single-Node Model:**

For many forecasting applications, a lumped single-node model suffices:

```
dT_gpu/dt = (P_gpu - (T_gpu - T_ambient) / R_total) / C_total
```

**Biot Number Validity Check:**

The lumped model is valid when the Biot number is small:

```
Bi = h * L_c / k_material < 0.1
```

where L_c is the characteristic length and k is thermal conductivity. For typical GPU packages, this condition is generally satisfied for steady-state analysis.

### 2.3 GPU Temperature-Power Relationship

The relationship between GPU temperature and power consumption is bidirectional:

**Power drives temperature:**
```
T_steady_state = T_ambient + R_thermal * P_gpu
```

**Temperature affects power (through leakage current):**
```
P_leakage(T) = P_leakage(T_ref) * exp(alpha_leak * (T - T_ref))
```

where alpha_leak is typically 0.01-0.02 per degree C for modern semiconductors.

**Thermal Throttling Thresholds:**

Modern NVIDIA GPUs implement multiple thermal thresholds:

| Threshold | Typical Value | Action |
|-----------|---------------|--------|
| **Idle** | 30-45 C | Normal operation, minimum clocks |
| **Normal Load** | 60-80 C | Full boost clocks available |
| **Throttle Start** | 83-84 C | Clock speed begins to reduce |
| **Throttle Aggressive** | 90-95 C | Significant clock reduction |
| **Critical/Shutdown** | 95-100 C | Emergency shutdown, system protection |

The throttling behavior can be modeled as:

```
effective_frequency = f_max * min(1.0, max(0.0, (T_critical - T_gpu) / (T_critical - T_throttle_start)))
```

### 2.4 Thermal Model for PINN Integration

```python
import torch
import torch.nn as nn

class ThermalRCModel(nn.Module):
    """
    Physics-informed thermal model for GPU temperature prediction.
    Implements the thermal RC network equation:
        dT/dt = (P - k*(T - T_ambient)) / C
    """
    def __init__(self):
        super().__init__()
        # Learnable thermal parameters (physically constrained)
        self.R_thermal = nn.Parameter(torch.tensor(0.5))  # K/W
        self.C_thermal = nn.Parameter(torch.tensor(100.0))  # J/K
        self.T_ambient = nn.Parameter(torch.tensor(25.0))  # C

    def forward(self, t, P_gpu, T_initial):
        """
        Compute GPU temperature over time given power input.

        Args:
            t: time tensor
            P_gpu: GPU power consumption (W)
            T_initial: initial GPU temperature (C)
        """
        # Ensure physical constraints
        R = torch.abs(self.R_thermal)  # R > 0
        C = torch.abs(self.C_thermal)  # C > 0

        tau = R * C  # thermal time constant

        # Analytical solution of the thermal ODE
        T_steady = self.T_ambient + R * P_gpu
        T = T_steady + (T_initial - T_steady) * torch.exp(-t / tau)

        return T

    def physics_loss(self, t, P_gpu, T_predicted):
        """
        Compute physics residual: dT/dt - (P - k*(T-T_amb))/C = 0
        """
        R = torch.abs(self.R_thermal)
        C = torch.abs(self.C_thermal)

        # Compute dT/dt via autograd
        dT_dt = torch.autograd.grad(
            T_predicted, t,
            grad_outputs=torch.ones_like(T_predicted),
            retain_graph=True, create_graph=True
        )[0]

        # Physics residual
        k = 1.0 / R
        residual = dT_dt - (P_gpu - k * (T_predicted - self.T_ambient)) / C

        return torch.mean(residual ** 2)
```

---

## 3. Power Modeling

### 3.1 GPU Power Consumption Model

GPU power consumption consists of static (leakage) and dynamic components:

```
P_total = P_static + P_dynamic
```

**Static Power (Leakage):**

```
P_static = P_leakage(T, V) = I_leak(T) * V_dd
```

Leakage current depends exponentially on temperature and voltage:

```
I_leak(T) = I_leak(T_ref) * exp(E_a / k * (1/T_ref - 1/T)) * (V_dd / V_ref)^alpha_v
```

where:
- E_a = activation energy (material-dependent)
- k = Boltzmann constant
- T_ref = reference temperature
- alpha_v = voltage scaling exponent

**Dynamic Power:**

The dynamic power follows the classic CMOS power equation:

```
P_dynamic = C_eff * V_dd^2 * f * alpha
```

where:
- C_eff = effective switching capacitance
- V_dd = supply voltage
- f = clock frequency
- alpha = activity factor (fraction of transistors switching)

### 3.2 Utilization-Based Power Model

For practical forecasting, a utilization-based model is commonly used:

```
P_gpu = P_idle + P_dynamic * U^alpha
```

where:
- P_idle = idle power consumption (typically 10-30W for modern GPUs)
- P_dynamic = maximum dynamic power (TDP - P_idle)
- U = GPU utilization (0 to 1)
- alpha = empirical exponent (typically 1.0-1.5 for GPUs)

**More detailed model accounting for frequency scaling:**

```
P_gpu = P_idle + C_eff * (f/f_max)^3 * U^alpha * P_dynamic_max
```

The cubic relationship `(f/f_max)^3` arises from voltage-frequency scaling:

```
V_dd proportional to f  (for modern DVFS)
P_dynamic proportional to V^2 * f proportional to f^3
```

### 3.3 Memory Bandwidth Effects

GPU memory subsystem contributes significantly to total power:

```
P_total = P_compute + P_memory + P_idle

P_compute = P_compute_max * U_compute^alpha * (f_core/f_max)^3

P_memory = P_memory_idle + P_memory_active * BW_utilization
```

where:
- BW_utilization = fraction of peak memory bandwidth being used
- P_memory_active = dynamic memory power (typically 20-60W for high-end GPUs)

**Combined Model:**

```python
def gpu_power_model(utilization, memory_bw_util, frequency_ratio,
                    params):
    """
    Comprehensive GPU power model.

    Args:
        utilization: GPU compute utilization (0-1)
        memory_bw_util: Memory bandwidth utilization (0-1)
        frequency_ratio: f/f_max (0-1)
        params: dict with P_idle, P_compute_max, P_memory_max, alpha, etc.
    """
    P_idle = params['P_idle']
    P_compute_max = params['P_compute_max']
    P_memory_max = params['P_memory_max']
    alpha = params['alpha']

    # Compute power with frequency scaling
    P_compute = P_compute_max * (frequency_ratio ** 3) * (utilization ** alpha)

    # Memory power (linear with bandwidth utilization)
    P_memory = P_memory_max * memory_bw_util

    # Temperature-dependent leakage (if temperature is known)
    T = params.get('temperature', 45)
    T_ref = params.get('T_ref', 25)
    leak_factor = np.exp(0.015 * (T - T_ref))

    P_static = P_idle * leak_factor

    return P_static + P_compute + P_memory
```

### 3.4 Voltage-Frequency Scaling

Modern GPUs use Dynamic Voltage and Frequency Scaling (DVFS) to balance performance and power:

```
P proportional to V^2 * f

Since V proportional to f (for modern process nodes):
P proportional to f^3
```

This means:
- Reducing frequency by 50% reduces power by approximately 87.5%
- Reducing frequency by 20% reduces power by approximately 49%

**DVFS States (Example for NVIDIA A100):**

| State | Frequency | Voltage | Power |
|-------|-----------|---------|-------|
| P0 (max boost) | 1410 MHz | ~0.85V | 400W |
| P1 (base) | 1095 MHz | ~0.75V | 250W |
| P2 (reduced) | 800 MHz | ~0.65V | 120W |
| P3 (idle) | 300 MHz | ~0.55V | 30W |

### 3.5 Physics-Informed Power Model

```python
class PhysicsInformedPowerModel(nn.Module):
    """
    Neural network with physics-informed constraints for GPU power prediction.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()

        # Data-driven component
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # inputs: util, mem_bw, freq, temp
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Learnable physics parameters
        self.P_idle = nn.Parameter(torch.tensor(25.0))
        self.alpha = nn.Parameter(torch.tensor(1.2))
        self.leak_coeff = nn.Parameter(torch.tensor(0.015))

    def forward(self, utilization, mem_bw, freq_ratio, temperature):
        """
        Predict GPU power with physics constraints.
        """
        # Physics-based baseline
        P_physics = self.P_idle + 350.0 * (freq_ratio ** 3) * (utilization ** self.alpha)

        # Data-driven residual correction
        inputs = torch.stack([utilization, mem_bw, freq_ratio, temperature], dim=-1)
        residual = self.net(inputs)

        # Combined prediction
        P_total = P_physics + residual

        return P_total

    def physics_constraints_loss(self, P_pred, utilization, temperature):
        """
        Enforce physical constraints on power predictions.
        """
        losses = {}

        # 1. Boundedness: P_idle <= P <= TDP
        losses['bounded_below'] = torch.mean(torch.relu(self.P_idle - P_pred) ** 2)
        losses['bounded_above'] = torch.mean(torch.relu(P_pred - 400.0) ** 2)  # TDP=400W

        # 2. Monotonicity: higher utilization -> higher power
        du = utilization[1:] - utilization[:-1]
        dP = P_pred[1:] - P_pred[:-1]
        losses['monotonicity'] = torch.mean(torch.relu(-du * dP) ** 2)

        # 3. Temperature-leakage relationship
        # Higher temperature should increase idle power
        T_effect = torch.exp(self.leak_coeff * (temperature - 25.0))
        losses['thermal_leakage'] = torch.mean((self.P_idle * T_effect - self.P_idle) ** 2)

        return losses
```

---

## 4. Physics-Informed Loss Functions

### 4.1 Conservation Laws as Constraints

Energy conservation is a fundamental constraint for power systems:

```
P_generation = P_consumption + P_losses + dE_stored/dt
```

**Implementation as loss term:**

```python
def conservation_loss(P_gen, P_load, P_losses, dE_dt):
    """
    Enforce energy conservation: P_gen = P_load + P_losses + dE_dt
    """
    residual = P_gen - P_load - P_losses - dE_dt
    return torch.mean(residual ** 2)
```

**Thermal energy conservation:**

```
P_input = P_dissipated + dU_thermal/dt
```

where U_thermal = C_thermal * T is the stored thermal energy.

### 4.2 Monotonicity Constraints

For GPU power modeling, higher utilization must yield higher power consumption:

```
dP/dU >= 0  (power is non-decreasing with utilization)
```

**Implementation:**

```python
def monotonicity_loss(model, x, feature_idx=0):
    """
    Enforce monotonicity: partial f / partial x_i >= 0
    """
    x.requires_grad_(True)
    y = model(x)

    dy_dx = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True, create_graph=True
    )[0]

    # Penalize negative gradients
    partial = dy_dx[:, feature_idx]
    violation = torch.relu(-partial)  # only penalize when dy/dx < 0

    return torch.mean(violation ** 2)
```

**Architectural approach (monotonic networks):**

```python
class MonotonicNetwork(nn.Module):
    """
    Neural network with guaranteed monotonicity in specified inputs.
    Uses non-negative weights for monotonic features.
    """
    def __init__(self, input_dim, hidden_dim, monotonic_features):
        super().__init__()
        self.monotonic_features = monotonic_features

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = torch.tanh(self.layer1(x))

        # Apply non-negative constraint to monotonic feature weights
        w = self.layer2.weight.clone()
        for idx in self.monotonic_features:
            w[:, idx] = torch.abs(w[:, idx])
        h = torch.tanh(F.linear(h, w, self.layer2.bias))

        return self.layer3(h)
```

### 4.3 Boundedness Constraints

Power output must remain within physical bounds:

```
P_idle <= P_predicted <= P_TDP
```

**Implementation:**

```python
def boundedness_loss(P_pred, P_min, P_max):
    """
    Enforce: P_min <= P_pred <= P_max
    """
    lower_violation = torch.relu(P_min - P_pred) ** 2
    upper_violation = torch.relu(P_pred - P_max) ** 2

    return torch.mean(lower_violation + upper_violation)


def boundedness_loss_soft(P_pred, P_min, P_max, margin=0.05):
    """
    Soft boundedness with safety margin.
    Penalizes predictions near boundaries.
    """
    range_P = P_max - P_min
    soft_min = P_min + margin * range_P
    soft_max = P_max - margin * range_P

    lower_violation = torch.relu(soft_min - P_pred) ** 2
    upper_violation = torch.relu(P_pred - soft_max) ** 2

    return torch.mean(lower_violation + upper_violation)
```

**Architectural approach (output transformation):**

```python
def bounded_output(raw_output, P_min, P_max):
    """
    Transform unbounded network output to [P_min, P_max] range.
    Uses sigmoid to guarantee bounds.
    """
    return P_min + (P_max - P_min) * torch.sigmoid(raw_output)
```

### 4.4 Smoothness Constraints

Physical quantities like power consumption cannot change instantaneously:

```
|dP/dt| <= ramp_rate_max
```

**Implementation:**

```python
def smoothness_loss(P_pred, t, max_ramp_rate=None):
    """
    Enforce smoothness: penalize large temporal gradients.
    """
    dP_dt = torch.autograd.grad(
        P_pred, t,
        grad_outputs=torch.ones_like(P_pred),
        retain_graph=True, create_graph=True
    )[0]

    if max_ramp_rate is not None:
        # Hard ramp rate limit
        violation = torch.relu(torch.abs(dP_dt) - max_ramp_rate)
        return torch.mean(violation ** 2)
    else:
        # Soft smoothness penalty (minimize second derivative)
        d2P_dt2 = torch.autograd.grad(
            dP_dt, t,
            grad_outputs=torch.ones_like(dP_dt),
            retain_graph=True, create_graph=True
        )[0]
        return torch.mean(d2P_dt2 ** 2)


def temporal_consistency_loss(P_pred_t, P_pred_t_prev, delta_t, max_change_rate):
    """
    Ensure power doesn't change too fast between consecutive predictions.
    """
    actual_rate = torch.abs(P_pred_t - P_pred_t_prev) / delta_t
    violation = torch.relu(actual_rate - max_change_rate)
    return torch.mean(violation ** 2)
```

### 4.5 Composite Physics-Informed Loss Function

```python
class CompositePhysicsLoss(nn.Module):
    """
    Comprehensive physics-informed loss function for energy forecasting.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Loss weights (can be learned)
        self.w_data = nn.Parameter(torch.tensor(config.get('w_data', 1.0)))
        self.w_conservation = nn.Parameter(torch.tensor(config.get('w_conservation', 0.1)))
        self.w_monotonicity = nn.Parameter(torch.tensor(config.get('w_monotonicity', 0.05)))
        self.w_boundedness = nn.Parameter(torch.tensor(config.get('w_boundedness', 0.1)))
        self.w_smoothness = nn.Parameter(torch.tensor(config.get('w_smoothness', 0.01)))

    def forward(self, model, inputs, targets, physics_params):
        """
        Compute composite physics-informed loss.
        """
        # Data loss
        predictions = model(inputs)
        loss_data = F.mse_loss(predictions, targets)

        # Conservation loss
        loss_conservation = conservation_loss(
            predictions, physics_params['P_load'],
            physics_params['P_losses'], physics_params['dE_dt']
        )

        # Monotonicity loss
        loss_monotonicity = monotonicity_loss(
            model, inputs, feature_idx=0  # utilization is feature 0
        )

        # Boundedness loss
        loss_boundedness = boundedness_loss(
            predictions,
            physics_params['P_min'],
            physics_params['P_max']
        )

        # Smoothness loss
        loss_smoothness = smoothness_loss(
            predictions, physics_params['t']
        )

        # Total loss with learnable weights
        total = (
            torch.exp(-self.w_data) * loss_data + self.w_data +
            torch.exp(-self.w_conservation) * loss_conservation + self.w_conservation +
            torch.exp(-self.w_monotonicity) * loss_monotonicity + self.w_monotonicity +
            torch.exp(-self.w_boundedness) * loss_boundedness + self.w_boundedness +
            torch.exp(-self.w_smoothness) * loss_smoothness + self.w_smoothness
        )

        return total, {
            'data': loss_data.item(),
            'conservation': loss_conservation.item(),
            'monotonicity': loss_monotonicity.item(),
            'boundedness': loss_boundedness.item(),
            'smoothness': loss_smoothness.item()
        }
```

---

## 5. Hybrid Models

### 5.1 Combining Data-Driven + Physics-Based

Hybrid models leverage the strengths of both approaches:

| Component | Strengths | Weaknesses |
|-----------|-----------|------------|
| **Physics-based** | Interpretable, extrapolates well, physically consistent | May be oversimplified, requires domain expertise |
| **Data-driven** | Captures complex patterns, adapts to data | Requires large datasets, may violate physics |
| **Hybrid** | Best of both worlds | More complex to train and tune |

**Architecture Patterns:**

```
Pattern 1: Physics-Enhanced Input
  Input -> [Physics Features] -> Neural Network -> Prediction

Pattern 2: Physics-Constrained Output
  Input -> Neural Network -> [Physics Correction] -> Prediction

Pattern 3: Parallel Combination
  Input -> Physics Model -----> \
                                 + -> Weighted Sum -> Prediction
  Input -> Neural Network -----> /

Pattern 4: Sequential (Residual)
  Input -> Physics Model -> Physics Prediction -> \
                                                    + -> Final Prediction
  Input + Physics Pred -> Neural Network -> Residual -> /
```

### 5.2 Residual Learning

The residual learning approach models the difference between a physics-based prediction and reality:

```
P_predicted = P_physics(x) + R_nn(x)
```

where R_nn is the neural network that learns the physics residual (errors in the simplified physics model).

**Advantages:**
- Physics model provides a strong baseline
- Neural network only needs to learn the correction
- More data-efficient than learning from scratch
- Predictions remain close to physical reality

**Implementation:**

```python
class ResidualPhysicsModel(nn.Module):
    """
    Hybrid model: physics baseline + neural network residual.
    """
    def __init__(self, physics_model, residual_net):
        super().__init__()
        self.physics_model = physics_model  # Fixed or learnable physics
        self.residual_net = residual_net    # Neural network

    def forward(self, x):
        # Physics-based baseline prediction
        with torch.no_grad():
            p_physics = self.physics_model(x)

        # Neural network residual correction
        residual = self.residual_net(x)

        # Combined prediction
        return p_physics + residual

    def physics_loss(self, x, y_true):
        """
        Loss that encourages the residual to be small
        (physics model should be mostly correct).
        """
        p_physics = self.physics_model(x)
        residual = self.residual_net(x)

        # Data fitting loss
        loss_data = F.mse_loss(p_physics + residual, y_true)

        # Regularization: penalize large residuals (prefer physics)
        loss_residual_reg = torch.mean(residual ** 2)

        return loss_data + 0.1 * loss_residual_reg


class GPUPhysicsBaseline:
    """
    Simple physics model for GPU power as baseline.
    """
    def __init__(self, P_idle=25.0, P_max=375.0, alpha=1.2):
        self.P_idle = P_idle
        self.P_max = P_max
        self.alpha = alpha

    def predict(self, utilization, frequency_ratio=1.0):
        return self.P_idle + self.P_max * (frequency_ratio ** 3) * (utilization ** self.alpha)
```

### 5.3 Multi-Task with Physics Auxiliary Tasks

Multi-task learning improves generalization by sharing representations across related tasks:

```python
class MultiTaskPhysicsModel(nn.Module):
    """
    Multi-task model with physics auxiliary tasks.
    Main task: power forecasting
    Auxiliary tasks: temperature prediction, efficiency estimation
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Task-specific heads
        self.power_head = nn.Linear(hidden_dim, 1)
        self.temperature_head = nn.Linear(hidden_dim, 1)
        self.efficiency_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_features = self.shared(x)

        power = self.power_head(shared_features)
        temperature = self.temperature_head(shared_features)
        efficiency = torch.sigmoid(self.efficiency_head(shared_features))  # 0-1

        return power, temperature, efficiency

    def multi_task_loss(self, x, y_power, y_temp, y_eff):
        """
        Multi-task loss with physics constraints.
        """
        pred_power, pred_temp, pred_eff = self.forward(x)

        # Data losses for each task
        loss_power = F.mse_loss(pred_power, y_power)
        loss_temp = F.mse_loss(pred_temp, y_temp)
        loss_eff = F.mse_loss(pred_eff, y_eff)

        # Physics coupling: power and temperature should be related
        # T = T_ambient + R_thermal * P
        R_thermal = 0.1  # K/W (could be learned)
        T_ambient = 25.0
        predicted_temp_from_power = T_ambient + R_thermal * pred_power.detach()
        loss_physics_coupling = F.mse_loss(pred_temp, predicted_temp_from_power)

        # Physics constraint: efficiency should decrease at extremes
        # (batteries are less efficient at very low/high SOC)
        loss_eff_physics = torch.mean(
            torch.relu(pred_eff - 0.95) ** 2 +  # efficiency <= 95%
            torch.relu(0.85 - pred_eff) ** 2     # efficiency >= 85%
        )

        total_loss = (
            loss_power +
            0.3 * loss_temp +
            0.2 * loss_eff +
            0.1 * loss_physics_coupling +
            0.05 * loss_eff_physics
        )

        return total_loss, {
            'power': loss_power.item(),
            'temperature': loss_temp.item(),
            'efficiency': loss_eff.item(),
            'physics_coupling': loss_physics_coupling.item()
        }
```

### 5.4 Physics-Informed Transfer Learning

```python
class PhysicsInformedTransferModel(nn.Module):
    """
    Transfer learning with physics constraints for new GPU architectures.
    Pre-train on source GPU, fine-tune on target GPU with physics constraints.
    """
    def __init__(self, pretrained_model):
        super().__init__()
        self.backbone = pretrained_model  # Pre-trained feature extractor

        # Replace final layer for new GPU architecture
        self.new_head = nn.Linear(64, 1)

        # Physics parameters (GPU-specific, fine-tuned)
        self.P_idle = nn.Parameter(torch.tensor(30.0))
        self.TDP = nn.Parameter(torch.tensor(400.0))

    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.new_head(features)

    def fine_tune_loss(self, x, y_true):
        """
        Fine-tuning loss with physics constraints for new GPU.
        """
        predictions = self.forward(x)
        loss_data = F.mse_loss(predictions, y_true)

        # Physics constraints (same as before, but with new GPU params)
        loss_bounds = (
            torch.mean(torch.relu(self.P_idle - predictions) ** 2) +
            torch.mean(torch.relu(predictions - self.TDP) ** 2)
        )

        return loss_data + 0.1 * loss_bounds
```

---

## 6. Battery Modeling

### 6.1 Tesla Megapack Specifications

The Tesla Megapack is a large-scale battery energy storage system designed for grid applications.

**Specifications by Model:**

| Spec | Megapack | Megapack 2 | Megapack 2 XL | Megapack 3 |
|------|----------|------------|---------------|------------|
| **Energy Capacity** | 2.6 MWh | 3.854 MWh | 3.916 MWh | ~5 MWh |
| **Power Output** | 1 MW | 1.284 MW | 1.927 MW | TBD |
| **Round-Trip Efficiency** | ~90% | 92.0% | 85-90% | 91% |
| **Weight** | 56,000 lb | 67,200 lb | 84,000 lb | 86,000 lb |
| **Cycle Life** | ~3,000 | ~3,000 | 3,000-5,000 | >10,000 |
| **Battery Chemistry** | NMC | LFP | LFP | LFP |

**Key Features:**
- Pre-assembled with battery modules, bi-directional inverters, thermal management, and controls
- Thermal management using ethylene glycol/water coolant mixture
- 15-year warranty (extendable to 20 years)
- Container-sized units with twistlock fittings for automated handling
- LFP (Lithium Iron Phosphate) chemistry in newer versions for improved safety

**Notable Deployments:**
- Victorian Big Battery (Geelong, Australia): 300 MW / 450 MWh
- Moss Landing (California): 182.5 MW / 730 MWh across 256 Megapacks
- Ventura County (California): 100 MW / 400 MWh, replacing a natural-gas peaker plant

### 6.2 State of Charge (SOC) Dynamics

SOC quantifies the remaining capacity in a battery, expressed as a percentage (0% = empty, 100% = full).

**Coulomb Counting (Current Integration):**

```
SOC(t) = SOC(t_0) + (1 / C_nominal) * integral(t_0 to t) I(tau) dtau
```

where:
- C_nominal = nominal battery capacity (Ah)
- I(t) = current at time t (positive for discharge, negative for charge)

**Discrete-Time Form:**

```
SOC(k+1) = SOC(k) + (I(k) * dt) / (C_nominal * eta_c)
```

where eta_c is the coulombic efficiency.

**Voltage-Based Estimation:**

The relationship between OCV (Open Circuit Voltage) and SOC is chemistry-dependent:

- **NMC (Nickel-Manganese-Cobalt):** Relatively linear OCV-SOC relationship, amenable to voltage-based estimation
- **LFP (Lithium Iron Phosphate):** Flat voltage plateau makes voltage-based estimation unreliable

**Kalman Filter Approach:**

A Kalman filter dynamically weights voltage and current measurements:

```python
class BatterySOCEstimator:
    """
    Extended Kalman Filter for battery SOC estimation.
    """
    def __init__(self, C_nominal, R_internal):
        self.C_nominal = C_nominal
        self.R_internal = R_internal

        # State: [SOC, V_polarization]
        self.x = np.array([0.5, 0.0])  # initial state
        self.P = np.eye(2) * 0.01  # initial covariance

        # Process noise
        self.Q = np.diag([1e-6, 1e-4])

        # Measurement noise
        self.R = np.array([[1e-3]])

    def ocv_from_soc(self, soc):
        """
        Open circuit voltage as function of SOC (lookup table).
        """
        # Simplified polynomial fit for NMC chemistry
        return 3.0 + 1.2 * soc - 0.5 * soc**2 + 0.3 * soc**3

    def predict(self, I, dt):
        """
        State prediction step.
        """
        soc = self.x[0]
        v_pol = self.x[1]

        # SOC dynamics
        dsoc = -I * dt / (self.C_nominal * 3600)
        # Polarization dynamics (RC model)
        tau = 10.0  # polarization time constant
        dv_pol = -v_pol / tau + I * self.R_internal / tau * dt

        self.x[0] = soc + dsoc
        self.x[1] = v_pol + dv_pol

        # State transition matrix
        F = np.array([
            [1, 0],
            [0, 1 - dt/tau]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, V_measured, I):
        """
        Measurement update step.
        """
        soc = self.x[0]
        v_pol = self.x[1]

        # Predicted terminal voltage
        V_predicted = self.ocv_from_soc(soc) - v_pol - I * self.R_internal

        # Innovation
        innovation = V_measured - V_predicted

        # Observation matrix (linearized)
        docv_dsoc = 1.2 - soc + 0.9 * soc**2  # derivative of OCV function
        H = np.array([[docv_dsoc, -1]])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K.flatten() * innovation
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.x[0]  # return SOC
```

### 6.3 Charge/Discharge Efficiency Curves

Battery efficiency varies with multiple factors:

**Coulombic Efficiency:**
```
eta_coulombic = Ah_out / Ah_in (typically >99% for Li-ion)
```

**Energy Efficiency:**
```
eta_energy = Wh_out / Wh_in (typically 85-95%)
```

**Efficiency Dependencies:**

```python
def battery_efficiency(soc, c_rate, temperature):
    """
    Model battery efficiency as function of SOC, C-rate, and temperature.

    Args:
        soc: state of charge (0-1)
        c_rate: charge/discharge rate (C)
        temperature: battery temperature (C)
    """
    # Base efficiency (varies with SOC)
    # Lower at extremes (very low/high SOC)
    eta_soc = 0.95 - 0.1 * (soc - 0.5)**2

    # C-rate penalty (higher rates = lower efficiency)
    eta_crate = 1.0 - 0.05 * c_rate

    # Temperature effect (optimal around 25-35C)
    eta_temp = 1.0 - 0.005 * (temperature - 30)**2

    # Combined efficiency
    eta_total = eta_soc * eta_crate * eta_temp

    return np.clip(eta_total, 0.7, 0.99)


def charge_efficiency(soc, c_rate, temperature):
    """
    Charge efficiency (slightly lower than discharge due to overpotential).
    """
    eta_base = battery_efficiency(soc, c_rate, temperature)
    return eta_base * 0.98  # 2% additional loss during charging


def discharge_efficiency(soc, c_rate, temperature):
    """
    Discharge efficiency.
    """
    return battery_efficiency(soc, c_rate, temperature)
```

**Efficiency Curve Characteristics:**

| Factor | Effect on Efficiency |
|--------|---------------------|
| SOC near 0% or 100% | Reduced efficiency (higher internal resistance) |
| High C-rate | Reduced efficiency (I^2R losses dominate) |
| Low temperature (<10 C) | Significantly reduced efficiency |
| Optimal temperature (25-35 C) | Peak efficiency |
| High temperature (>45 C) | Reduced efficiency, accelerated degradation |

### 6.4 Optimal Battery Dispatch Strategies

**Objective Functions:**

```python
def dispatch_objective(schedule, prices, load, battery_params):
    """
    Objective: minimize electricity cost while respecting constraints.

    Args:
        schedule: charge/discharge power at each time step (kW)
        prices: electricity price at each time step ($/kWh)
        load: power demand at each time step (kW)
        battery_params: dict with capacity, max_power, efficiency, etc.
    """
    total_cost = 0.0
    soc = battery_params['initial_soc']

    for t in range(len(schedule)):
        p_battery = schedule[t]  # positive = discharge, negative = charge

        # Grid power = load + charging - discharging
        p_grid = load[t] + max(0, -p_battery) - max(0, p_battery) * battery_params['eta_discharge']

        # Cost at this time step
        total_cost += p_grid * prices[t] * dt

        # Update SOC
        if p_battery > 0:  # discharge
            soc -= p_battery * dt / (battery_params['capacity'] * battery_params['eta_discharge'])
        else:  # charge
            soc += abs(p_battery) * dt * battery_params['eta_charge'] / battery_params['capacity']

    return total_cost


def dispatch_constraints(schedule, battery_params, soc_initial):
    """
    Constraint functions for battery dispatch optimization.
    """
    constraints = []

    # SOC bounds: 0.1 <= SOC <= 0.9 (to extend battery life)
    constraints.append({'type': 'ineq', 'fun': lambda s: soc_from_schedule(s) - 0.1})
    constraints.append({'type': 'ineq', 'fun': lambda s: 0.9 - soc_from_schedule(s)})

    # Power bounds: -P_max <= P <= P_max
    constraints.append({'type': 'ineq', 'fun': lambda s: battery_params['max_power'] - np.abs(s)})

    # Ramp rate: |dP/dt| <= ramp_max
    constraints.append({'type': 'ineq', 'fun': lambda s: battery_params['ramp_max'] - np.abs(np.diff(s))})

    # SOC dynamics consistency
    constraints.append({'type': 'eq', 'fun': lambda s: soc_terminal - soc_initial})

    return constraints
```

**Dispatch Algorithms:**

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Linear Programming** | Fast, globally optimal | Assumes linearity |
| **MILP** | Handles discrete decisions | Computationally expensive |
| **Dynamic Programming** | Multi-stage optimization | Curse of dimensionality |
| **Model Predictive Control** | Handles uncertainty, rolling horizon | Requires forecasts |
| **Reinforcement Learning** | Adaptive, learns from experience | Training instability |

---

## 7. Grid Stability Constraints

### 7.1 Frequency Stability (60 Hz Nominal)

The North American power grid operates at a nominal frequency of 60 Hz. Frequency deviations indicate supply-demand imbalance:

```
df/dt proportional to (P_generation - P_load) / H_total
```

where H_total is the total system inertia.

**Frequency Response Model:**

```python
def frequency_dynamics(P_gen, P_load, H_system, D_load, t):
    """
    Simplified frequency dynamics model.

    Args:
        P_gen: total generation (MW)
        P_load: total load (MW)
        H_system: system inertia constant (s)
        D_load: load damping coefficient (MW/Hz)
        t: time array
    """
    # Power imbalance
    delta_P = P_gen - P_load  # MW

    # Frequency deviation (swing equation approximation)
    # 2H * df/dt = delta_P - D * delta_f
    # Steady-state: delta_f = delta_P / D

    # Time constant
    tau = 2 * H_system / D_load

    # Frequency response
    delta_f_ss = delta_P / D_load  # steady-state deviation
    delta_f = delta_f_ss * (1 - np.exp(-t / tau))

    return 60.0 + delta_f  # Hz


def frequency_constraint(P_gen_forecast, P_load_forecast, H=5.0, D=1.0, max_deviation=0.5):
    """
    Check if frequency stays within acceptable bounds.

    Returns: True if frequency constraint is satisfied
    """
    delta_P = abs(P_gen_forecast - P_load_forecast)
    steady_state_deviation = delta_P / D

    return steady_state_deviation <= max_deviation
```

**Frequency Stability as Model Constraint:**

```python
def frequency_stability_loss(P_gen_pred, P_load_pred, H=5.0, D=1.0, max_dev=0.5):
    """
    Loss term penalizing predictions that would cause frequency instability.
    """
    delta_P = P_gen_pred - P_load_pred
    expected_deviation = torch.abs(delta_P) / D

    # Penalize deviations beyond acceptable range
    violation = torch.relu(expected_deviation - max_dev)

    return torch.mean(violation ** 2)
```

### 7.2 Ramp Rate Limits

Ramp rate limits constrain how fast power output can change:

```
|dP/dt| <= ramp_rate_max
```

**Typical Values:**

| Resource | Ramp Rate Limit |
|----------|-----------------|
| Gas Turbine (peaker) | 10-20 MW/min |
| Combined Cycle | 5-10 MW/min |
| Coal | 1-5 MW/min |
| Battery Storage | 100+ MW/min (very fast) |
| Solar (with clouds) | Variable, can be steep |

**Implementation:**

```python
def ramp_rate_constraint(P_pred, dt, max_ramp_rate):
    """
    Enforce ramp rate limits on predicted power output.

    Args:
        P_pred: predicted power sequence (MW)
        dt: time step (minutes)
        max_ramp_rate: maximum allowed ramp (MW/min)
    """
    ramp_rates = torch.abs(P_pred[1:] - P_pred[:-1]) / dt
    violations = torch.relu(ramp_rates - max_ramp_rate)

    return torch.mean(violations ** 2)
```

### 7.3 Power Factor Requirements

Most grid operators require power factor between 0.95 leading and 0.95 lagging.

```
Power Factor = P / S = cos(phi)
```

where:
- P = real power (W)
- S = apparent power (VA)
- phi = phase angle between voltage and current

**Reactive Power:**

```
Q = P * tan(phi)
S = P / cos(phi) = sqrt(P^2 + Q^2)
```

**Constraint Implementation:**

```python
def power_factor_constraint(P_pred, Q_pred, min_pf=0.95):
    """
    Enforce power factor constraints.

    Args:
        P_pred: predicted real power
        Q_pred: predicted reactive power
        min_pf: minimum power factor (default 0.95)
    """
    S = torch.sqrt(P_pred**2 + Q_pred**2)
    pf = P_pred / (S + 1e-8)  # avoid division by zero

    # Penalize power factor below minimum
    violation = torch.relu(min_pf - pf)

    return torch.mean(violation ** 2)
```

### 7.4 Comprehensive Grid Stability Loss

```python
class GridStabilityLoss(nn.Module):
    """
    Comprehensive grid stability constraints as loss functions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, P_gen, P_load, Q_gen=None, schedule=None):
        losses = {}

        # 1. Power balance (generation = load + losses)
        losses['power_balance'] = torch.mean((P_gen - P_load) ** 2)

        # 2. Frequency stability
        H = self.config.get('inertia_constant', 5.0)
        D = self.config.get('damping', 1.0)
        max_dev = self.config.get('max_freq_deviation', 0.5)
        delta_P = torch.abs(P_gen - P_load)
        freq_violation = torch.relu(delta_P / D - max_dev)
        losses['frequency'] = torch.mean(freq_violation ** 2)

        # 3. Ramp rate limits
        max_ramp = self.config.get('max_ramp_rate', 10.0)  # MW/min
        dt = self.config.get('dt', 1.0)  # minutes
        if schedule is not None and len(schedule) > 1:
            ramps = torch.abs(schedule[1:] - schedule[:-1]) / dt
            ramp_violation = torch.relu(ramps - max_ramp)
            losses['ramp_rate'] = torch.mean(ramp_violation ** 2)

        # 4. Power factor
        if Q_gen is not None:
            min_pf = self.config.get('min_power_factor', 0.95)
            S = torch.sqrt(P_gen**2 + Q_gen**2)
            pf = P_gen / (S + 1e-8)
            pf_violation = torch.relu(min_pf - pf)
            losses['power_factor'] = torch.mean(pf_violation ** 2)

        # 5. Reserve margin (generation should exceed load by margin)
        reserve_margin = self.config.get('reserve_margin', 0.1)
        required_gen = P_load * (1 + reserve_margin)
        reserve_violation = torch.relu(required_gen - P_gen)
        losses['reserve'] = torch.mean(reserve_violation ** 2)

        return losses
```

---

## 8. Implementation Architecture

### 8.1 Complete Physics-Informed Energy Forecasting Pipeline

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class EnergyForecastingPINN(nn.Module):
    """
    Complete physics-informed neural network for energy/power forecasting.
    Combines GPU power modeling, thermal dynamics, battery storage, and grid constraints.
    """
    def __init__(self, config: Dict):
        super().__init__()

        # Model configuration
        self.config = config

        # GPU Power Model
        self.gpu_power_net = nn.Sequential(
            nn.Linear(5, 128),   # [util, mem_bw, freq, temp, time]
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Thermal Model
        self.thermal_net = nn.Sequential(
            nn.Linear(4, 64),    # [power, temp_prev, ambient, time]
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Battery Model
        self.battery_net = nn.Sequential(
            nn.Linear(5, 64),    # [soc, power_req, price, temp, time]
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)     # [charge_power, discharge_power]
        )

        # Physics parameters (learnable)
        self.P_idle = nn.Parameter(torch.tensor(25.0))
        self.alpha = nn.Parameter(torch.tensor(1.2))
        self.R_thermal = nn.Parameter(torch.tensor(0.1))
        self.C_thermal = nn.Parameter(torch.tensor(500.0))
        self.battery_capacity = nn.Parameter(torch.tensor(3900.0))  # kWh (Megapack)
        self.battery_efficiency = nn.Parameter(torch.tensor(0.92))

        # Loss weights
        self.loss_weights = nn.ParameterDict({
            'data': nn.Parameter(torch.tensor(1.0)),
            'physics': nn.Parameter(torch.tensor(0.1)),
            'thermal': nn.Parameter(torch.tensor(0.05)),
            'battery': nn.Parameter(torch.tensor(0.05)),
            'grid': nn.Parameter(torch.tensor(0.02))
        })

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all sub-models.

        Args:
            x: dict containing:
                - gpu_util: GPU utilization (0-1)
                - mem_bw: memory bandwidth utilization (0-1)
                - freq_ratio: frequency ratio (0-1)
                - temperature: current temperature
                - ambient_temp: ambient temperature
                - time: time tensor
                - soc: current battery state of charge
                - price: electricity price
        """
        outputs = {}

        # GPU Power prediction
        gpu_input = torch.stack([
            x['gpu_util'], x['mem_bw'], x['freq_ratio'],
            x['temperature'], x['time']
        ], dim=-1)
        outputs['gpu_power'] = self.gpu_power_net(gpu_input)

        # Physics-based power baseline
        physics_power = self.P_idle + 350.0 * (x['freq_ratio'] ** 3) * (x['gpu_util'] ** self.alpha)
        outputs['physics_power'] = physics_power
        outputs['residual_power'] = outputs['gpu_power'] - physics_power

        # Thermal prediction
        thermal_input = torch.stack([
            outputs['gpu_power'], x['temperature'],
            x['ambient_temp'], x['time']
        ], dim=-1)
        outputs['temperature_pred'] = self.thermal_net(thermal_input)

        # Battery dispatch
        battery_input = torch.stack([
            x['soc'], outputs['gpu_power'], x['price'],
            x['temperature'], x['time']
        ], dim=-1)
        battery_output = self.battery_net(battery_input)
        outputs['battery_charge'] = torch.sigmoid(battery_output[:, 0:1])
        outputs['battery_discharge'] = torch.sigmoid(battery_output[:, 1:2])

        return outputs

    def compute_losses(self, x: Dict, targets: Dict, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute all loss components.
        """
        loss_dict = {}

        # 1. Data loss
        loss_dict['data'] = F.mse_loss(outputs['gpu_power'], targets['power'])

        # 2. Physics residual loss
        residual = outputs['residual_power']
        loss_dict['physics'] = torch.mean(residual ** 2)

        # 3. Thermal physics loss
        # dT/dt = (P - k*(T-T_amb)) / C
        k = 1.0 / torch.abs(self.R_thermal)
        C = torch.abs(self.C_thermal)
        T_pred = outputs['temperature_pred']
        dT_dt = torch.autograd.grad(
            T_pred, x['time'],
            grad_outputs=torch.ones_like(T_pred),
            retain_graph=True, create_graph=True
        )[0]
        thermal_residual = dT_dt - (outputs['gpu_power'] - k * (T_pred - x['ambient_temp'])) / C
        loss_dict['thermal'] = torch.mean(thermal_residual ** 2)

        # 4. Battery SOC loss
        # SOC dynamics: dSOC/dt = (P_charge*eta - P_discharge/eta) / capacity
        dt = 1.0 / 3600  # 1 second in hours
        soc_change = (
            outputs['battery_charge'] * self.battery_efficiency -
            outputs['battery_discharge'] / self.battery_efficiency
        ) * dt / self.battery_capacity
        loss_dict['battery'] = F.mse_loss(x['soc'] + soc_change, targets['soc_next'])

        # 5. Grid stability loss
        total_gen = outputs['gpu_power'] + outputs['battery_discharge'] * self.battery_capacity
        total_load = targets['load'] + outputs['battery_charge'] * self.battery_capacity
        loss_dict['grid'] = torch.mean((total_gen - total_load) ** 2)

        # 6. Constraint violations
        loss_dict['boundedness'] = (
            torch.mean(torch.relu(self.P_idle - outputs['gpu_power']) ** 2) +
            torch.mean(torch.relu(outputs['gpu_power'] - 400.0) ** 2)
        )

        # Total loss
        total_loss = sum(
            torch.exp(-self.loss_weights[k]) * v + self.loss_weights[k]
            for k, v in loss_dict.items()
            if k in self.loss_weights
        )

        return total_loss, loss_dict

    def train_step(self, x: Dict, targets: Dict, optimizer: torch.optim.Optimizer):
        """
        Single training step.
        """
        optimizer.zero_grad()

        outputs = self.forward(x)
        total_loss, loss_dict = self.compute_losses(x, targets, outputs)

        total_loss.backward()
        optimizer.step()

        return total_loss.item(), loss_dict


# Example usage
config = {
    'P_idle': 25.0,
    'P_max': 400.0,
    'thermal_R': 0.1,
    'thermal_C': 500.0,
    'battery_capacity': 3900.0,
    'battery_efficiency': 0.92,
    'max_ramp_rate': 10.0,
    'reserve_margin': 0.1
}

model = EnergyForecastingPINN(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop would go here...
```

### 8.2 Recommended Libraries

| Library | Purpose | Link |
|---------|---------|------|
| **DeepXDE** | General-purpose PINN library | github.com/lululxvi/deepxde |
| **NVIDIA Modulus** | Production-grade PINNs | github.com/NVIDIA/modulus |
| **PyTorch** | Deep learning framework | pytorch.org |
| **PyPSA** | Power system analysis | pypsa.org |
| **PyBaMM** | Battery modeling | pybamm.org |

---

## 9. References

### Foundational PINN Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. **Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L.** (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440.

3. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E.** (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.

### GPU Power and Thermal Modeling

4. **Hong, S., & Kim, H.** (2010). An integrated GPU power and performance model. *Proceedings of the 37th Annual International Symposium on Computer Architecture (ISCA)*.

5. **Leng, J., Hetherington, T., ElTantawy, A., Gilani, S., Kim, N. S., & Reddi, V. J.** (2013). GPUWattch: Enabling energy optimizations in GPGPUs. *Proceedings of the 40th Annual International Symposium on Computer Architecture (ISCA)*.

6. **NVIDIA.** NVML API Reference. developer.nvidia.com/nvidia-management-library-nvml

### Battery Modeling

7. **Plett, G. L.** (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.

8. **Plett, G. L.** (2015). *Battery Management Systems, Volume II: Equivalent-Circuit Methods*. Artech House.

9. **Doyle, M., Fuller, T. F., & Newman, J.** (1993). Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell. *Journal of the Electrochemical Society*, 140(6), 1526-1533.

### Grid Stability

10. **Kundur, P.** (1994). *Power System Stability and Control*. McGraw-Hill.

11. **IEEE Standard 1547-2018.** Standard for Interconnection and Interoperability of Distributed Energy Resources with Associated Electric Power Systems Interfaces.

12. **NERC Reliability Standards.** North American Electric Reliability Corporation. nerc.com

### Tesla Megapack

13. **Tesla, Inc.** Megapack Specifications. tesla.com/megapack

14. **Tesla, Inc.** (2023-2024). Quarterly Reports and SEC Filings. ir.tesla.com

### Hybrid Physics-ML Models

15. **Karpatne, A., Atluri, G., Faghmous, J. H., Steinbach, M., Banerjee, A., Ganguly, A., ... & Kumar, V.** (2017). Theory-guided data science: A new paradigm for scientific discovery from data. *IEEE Transactions on Knowledge and Data Engineering*, 29(10), 2318-2331.

16. **Beucler, T., Pritchard, M., Yuval, J., Gupta, A., Peng, L., Rasp, S., ... & Yu, S.** (2021). Enforcing analytic constraints in neural networks emulating physical systems. *Physical Review Letters*, 126(9), 098302.

17. **Wang, S., Teng, Y., & Perdikaris, P.** (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055-A3081.

### Thermal Engineering Standards

18. **JEDEC JESD51 Series.** Thermal Measurement Standards. jedec.org

19. **Lasance, C. J.** (2003). Ten years of boundary-condition-independent compact thermal modeling of electronic parts: A review. *Heat Transfer Engineering*, 24(2), 1-14.

---

*Document generated for ENERGIVANU project. Last updated: 2026-05-27.*
