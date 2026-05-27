# GPU Power Demand Forecasting for ENERGIVANU

## Research Document: Predicting GPU Power Demand 10 Minutes Ahead

**Target Application:** xAI Colossus Supercomputer (100K+ H100 GPUs) with Tesla Megapack Battery Signaling

**Last Updated:** 2026-05-27

---

## Table of Contents

1. [PI-DLinear: Physics-Informed DLinear for GPU Power Forecasting](#1-pi-dlinear-physics-informed-dlinear-for-gpu-power-forecasting)
2. [GPU Power Modeling in Data Centers](#2-gpu-power-modeling-in-data-centers)
3. [xAI Colossus Infrastructure](#3-xai-colossus-infrastructure)
4. [NVIDIA DCGM: Data Center GPU Manager](#4-nvidia-dcgm-data-center-gpu-manager)
5. [Power Throttling and DVFS Effects](#5-power-throttling-and-dvfs-effects)
6. [Related Research Papers](#6-related-research-papers)
7. [Implementation Architecture for ENERGIVANU](#7-implementation-architecture-for-energivanu)

---

## 1. PI-DLinear: Physics-Informed DLinear for GPU Power Forecasting

**Paper:** "A Physics-Aware Framework for Short-Term GPU Power Forecasting of AI Data Centers"
**arXiv:** 2605.04074v1
**Authors:** Mohammad AlShaikh Saleh, Sanjay Chawla, Sertac Bayhan, Haitham Abu-Rub, Ali Ghrayeb
**License:** CC BY 4.0

### 1.1 Core Innovation

PI-DLinear is the first physics-informed DLinear model for GPU power forecasting. It embeds thermal physics (Newton's cooling law, thermal RC networks) directly into the loss function of a simple linear decomposition model, achieving superior accuracy to 16 transformer-based and non-transformer baselines while maintaining identical parameter count (96,160 params) and memory footprint (0.376 MB) as vanilla DLinear.

**Key insight:** GPUs convert nearly 99% of electrical energy into heat through transistor switching losses, resistive losses in interconnects, and leakage currents. This tight coupling between power and temperature enables physics-informed forecasting.

### 1.2 Thermal RC Network Model (Two-Node Coupled System)

The GPU system is modeled as a lumped-parameter thermal resistance-capacitance (RC) network with two thermal nodes:

**GPU Node (T_g):**
```
C_g * dT_g/dt = alpha * P - (T_g - T_a) / R_ga - (T_g - T_m) / R_gm
```

**Memory Node (T_m):**
```
C_m * dT_m/dt = (1 - alpha) * P - (T_m - T_a) / R_ma + (T_g - T_m) / R_gm
```

**Where:**
- `C_g, C_m > 0` = thermal capacitances (heat storage capacity, units: J/K)
- `R_ga, R_ma > 0` = thermal resistances from each component to ambient (K/W)
- `R_gm > 0` = thermal coupling resistance between GPU and memory nodes (K/W)
- `alpha in [0,1]` = latent power split parameter between GPU compute and memory
- `T_a` = ambient temperature (assumed constant at 27 deg C, the minimum observed)
- `P` = total power consumption (Watts)

**Physical interpretation:** Each equation is an energy balance. The left side is the rate of heat storage. The right side terms are: (1) power dissipated as heat in that node, (2) heat loss to ambient via Newton's cooling law (analogous to Ohm's law: delta_T = Q_dot * R_th), and (3) heat exchange between GPU and memory nodes.

### 1.3 Power Rate Constraint (Derived ODE)

By solving the GPU node ODE for P and differentiating, the authors derive a constraint on power rate of change:

```
dP/dt = (1/alpha) * [C_g * d^2 T_g/dt^2 + (1/R_ga) * dT_g/dt + (1/R_gm) * (dT_g/dt - dT_m/dt)]
```

This links the time derivatives of temperature measurements to the dynamics of power consumption, enabling the model to predict power trajectories that respect thermal physics.

### 1.4 Estimated RC Parameters

Fitted via Recursive Least Squares (RLS) on the MIT Supercloud training data:

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| C_g | 5.408 x 10^6 J/K | GPU thermal capacitance |
| C_m | 5.481 x 10^6 J/K | Memory thermal capacitance |
| R_ga | 2.037 x 10^-3 K/W | GPU-to-ambient thermal resistance |
| R_ma | 2.055 x 10^-3 K/W | Memory-to-ambient thermal resistance |
| R_gm | 6.064 x 10^-4 K/W | GPU-to-memory coupling resistance |
| T_a | 27 deg C | Ambient temperature |
| alpha | 0.5085 | Power split fraction (GPU vs memory) |

**Implementation note:** For a new deployment (e.g., H100 instead of V100), these parameters must be re-estimated from training data using RLS. The alpha parameter is particularly workload-dependent.

### 1.5 DLinear Architecture

DLinear (Zeng et al., 2023) decomposes time series into trend and seasonal/remainder components, then applies separate single-layer linear networks:

```
H_s = W_s * X_s  (seasonal/residual component)
H_t = W_t * X_t  (trend component)
y_hat = H_s + H_t
```

Where:
- `W_s in R^{T x L}` and `W_t in R^{T x L}` are learnable weight matrices
- `T` = prediction horizon, `L` = look-back window length
- Each covariate gets independent weights (not shared across channels)
- Trend is extracted via moving average kernel
- Seasonal = original - trend

**Why DLinear over Transformers:** The authors note DLinear's stability -- MAE varies only 0.1420 to 0.1439 and MSE 0.1556 to 0.1576 as history grows from 240 to 600 minutes. Transformers showed more variance and higher computational cost.

### 1.6 Loss Function with Physics Constraints

The total loss has three components:

```
L_total = lambda_u * L_Data + lambda_r * L_r + lambda_theta * L_throttle
```

#### Data Loss (MSE):
```
L_pred = (1/T) * sum_{k=1}^{T} (P_hat_{t+k} - P_{t+k})^2
```

#### Physics Residual Loss:
```
L_r = (1/N_r) * sum_{i=1}^{N_r} |P(x_r^i, t_r^i; Phi, phi)|^2
```

This enforces consistency between predicted power and observed temperatures T_g, T_m through the derived ODE constraints. The residual is computed using automatic differentiation of the surrogate model's predictions.

#### Power Throttling Constraint:

Throttling events are defined as "sudden power drops exceeding 15%." The constraint encodes: when utilization exceeds threshold theta_U (approx 90%), power should not increase.

```
L_throttle = L_high + L_stress

L_high = (1/(H-1)) * sum_{t: U_t > theta_U} max(0, delta_P_hat_t)^2

L_stress = (1/(H-1)) * sum_{t: U_t > theta_U, T_t^g > theta_T} max(0, delta_P_hat_t)^2
```

Where:
- `delta_P_hat_t = P_hat_{t+1} - P_hat_t` (predicted power change)
- `U_t = alpha * u_t^(g) + (1-alpha) * u_t^(m)` (weighted GPU-memory utilization)
- `theta_U` = utilization threshold (approximately 90%)
- `theta_T` = temperature threshold (95th percentile)
- `H` = prediction horizon

**Physical rationale:** At near-maximum utilization, the GPU is already at or near its power cap. Further increases in utilization should not predict higher power -- the model should predict throttling instead.

#### Self-Adaptive Weighting:

Weights are updated via gradient ascent in log-space:
```
eta_u = log(lambda_u), eta_r = log(lambda_r), eta_theta = log(lambda_theta)
Update: eta <- eta + gamma * grad_eta(L)
Weights clipped to [lambda_min, lambda_max] to prevent instability.
```

This eliminates manual hyperparameter tuning of the loss weights. The authors tested fixed lambda = 0.005 as a baseline and found self-adaptive weighting consistently better.

### 1.7 Input Features (5 covariates)

1. GPU utilization (%)
2. Memory utilization (%)
3. GPU temperature (deg C)
4. Memory temperature (deg C)
5. Power draw (Watts)

**Omitted features:** Memory used and memory free were excluded due to low observed correlation with power in the dataset.

### 1.8 Task Definition

Given historical data `X = {x_1, ..., x_L} in R^{L x C}`, predict:
```
y_hat_{t+1:t+T} = f_theta(X_{t-L+1:t}) in R^T
```

Where L = look-back window, T = prediction horizon, C = 5 covariates.

### 1.9 Dataset: MIT Supercloud

| Attribute | Value |
|-----------|-------|
| Source | Samsi et al., 2021 |
| Raw covariates | 7 (5 selected) |
| Timesteps | approximately 330,500 |
| Granularity | 1 minute |
| Duration | approximately 238 days (Feb-Oct 2021) |
| GPU type | NVIDIA Volta V100 |
| CPU | Intel Xeon Gold 6248 |
| Max aggregated GPU power | 45 kW across 448 GPUs |
| Per-GPU nominal power | approximately 250 W |

**Workload composition:**
- Vision: U-Net (1,431 jobs), VGG, ResNet, Inception
- LLMs: BERT (189 jobs), DistillBERT (172 jobs)
- GNNs: SchNet, DimeNetm, PNA, conv

**Preprocessing:** Min-Max normalization based on training set statistics; data aggregated by job ID and node at 1-minute granularity.

### 1.10 Forecasting Results

**Benchmark models (16 total):** Transformer, iTransformer, TimeXer, TiDE, TSMixer, Reformer, PatchTST, Nonstationary Transformer, LightTS, FiLM, FEDformer, Pyraformer, DLinear, Crossformer, NLinear, Linear

**Forecasting horizons tested:** L in {240, 360, 480, 600} minutes look-back; T in {5, 10, 20, 40, 80} minutes prediction

#### Key Results (T=240 look-back, averaged across prediction horizons):

| Model | MAE | MSE | MAPE | RMSE |
|-------|-----|-----|------|------|
| DLinear | 0.1420 | 0.1556 | 1.0403 | 0.3907 |
| FiLM | 0.1432 | 0.1571 | 1.0708 | 0.3925 |
| TiDE | 0.1422 | 0.1561 | 1.0628 | 0.3912 |
| iTransformer | 0.1481 | 0.1636 | 1.1323 | 0.4000 |
| Transformer | 0.1670 | 0.1722 | 1.2211 | 0.4095 |
| **PI-DLinear** | **0.1420** | **0.1546** | **1.0315** | **0.3895** |

#### Overall Improvement Ranges (averaged across all windows):
- MSE: 0.782% to 39.08% improvement over SOTA
- MAE: 0.993% to 51.82% improvement
- RMSE: 0.370% to 22.28% improvement

### 1.11 Throttle Prediction Results

- PI-DLinear improves throttle detection rate by **6.88% on average** across all configurations
- Best improvement: L=360, T=10: 85.03% vs 65.29% (DLinear), a 19.75% gain
- Near-perfect detection at L=480, H=80: 99.12% vs 96.27%
- Throttle RMSE improvement: **3.92%** average; best case reduced error from 1,256 W to 897 W
- Physics constraints most beneficial for prediction horizons T >= 10 minutes

### 1.12 Ablation Study

| Variant | 5min MAPE | 20min MAPE | 80min MAPE |
|---------|-----------|------------|------------|
| Derived ODE Solution only | 0.7409 | 0.9574 | 1.5252 |
| DLinear (no physics) | 0.7331 | 0.9606 | 1.5250 |
| PI-DLinear + Constant lambda | 0.7317 | 0.9561 | 1.5088 |
| **Self-Adaptive PI-DLinear** | **0.7309** | **0.9554** | **1.4987** |

**Key finding:** Self-adaptive weighting delivers best scores at every prediction length, with largest gains at longer horizons. The physics component helps most when extrapolating further into the future.

### 1.13 Computational Cost

| Model | Parameters | Time (s/epoch) | Memory (MB) |
|-------|-----------|-----------------|-------------|
| DLinear | 96,160 | 10.43 | 0.376 |
| PI-DLinear | 96,160 | 20.27 | 0.376 |
| FiLM | 12,923,662 | 271.38 | 49.30 |
| TiDE | 1,624,359 | 29.10 | 6.210 |

PI-DLinear is approximately 1.9x slower during training only. The physics-aware component adds compute overhead without increasing model size because it is only used during training optimization, not during inference.

### 1.14 Implementation Notes for ENERGIVANU

1. **Adaptation to H100:** The RC parameters were fitted to V100 GPUs. For H100 (700W TDP vs 250W), the thermal capacitances and resistances will be different. Must re-estimate using RLS on H100 telemetry data.

2. **Scale considerations:** The paper tested on 448 GPUs aggregating to 45 kW. ENERGIVANU targets 100,000+ GPUs at 70 MW. The model should work at aggregate level since thermal dynamics are per-GPU and can be summed.

3. **10-minute horizon is well within tested range:** The paper tested T=5, 10, 20, 40, 80 minutes. T=10 is directly supported and shows strong performance.

4. **Inference speed:** With only 96K parameters and a single linear layer per component, inference is essentially instant -- critical for real-time battery signaling.

5. **Feature collection:** Need DCGM or equivalent to collect: GPU utilization, memory utilization, GPU temperature, memory temperature, power draw -- all at 1-minute granularity.

---

## 2. GPU Power Modeling in Data Centers

### 2.1 Power Components of a GPU

GPU power consumption has three primary components:

**Dynamic Power (dominant):**
```
P_dynamic = C_eff * V^2 * f * alpha
```
Where:
- `C_eff` = effective switching capacitance
- `V` = supply voltage
- `f` = clock frequency
- `alpha` = activity factor (fraction of transistors switching)

**Static Power (leakage):**
```
P_static = V * I_leakage
```
Leakage current increases exponentially with temperature and voltage. At advanced process nodes (5nm for H100), static power is a significant fraction of total power.

**I/O Power:**
Power consumed by memory interfaces (HBM3), NVLink, PCIe. Scales with memory bandwidth utilization.

### 2.2 H100 SXM5 Power Characteristics

| Specification | Value |
|---------------|-------|
| TDP (Thermal Design Power) | 700W |
| Idle Power | approximately 75-100W |
| Typical Load Power | 600-700W |
| Memory (HBM3) | 80 GB at 3.35 TB/s |
| Max Junction Temperature | 83 deg C |
| Process Node | TSMC 4N |
| Transistors | 80 billion |
| FP64 TFLOPS | 34 |
| FP8 TFLOPS | 1,979 |
| Interconnect | NVLink 900 GB/s |

**DGX H100 system (8x H100 SXM5):** Up to approximately 10.2 kW total system power under full load.

### 2.3 Power-Utilization Relationship

GPU power does not scale linearly with utilization:

- **Idle (0% util):** approximately 75-100W (memory controllers, clock trees, leakage)
- **Light load (10-30%):** approximately 200-350W
- **Medium load (50-70%):** approximately 400-550W
- **Heavy load (80-100%):** approximately 550-700W
- **Memory-bound vs compute-bound:** Memory-intensive workloads (high HBM bandwidth) draw differently than compute-bound workloads (high FLOPS) at the same "utilization" percentage

**Important for modeling:** The relationship between `nvidia-smi` GPU utilization percentage and actual power is nonlinear and workload-dependent. A workload at 80% GPU util may draw 550W or 650W depending on whether it is memory-bound or compute-bound.

### 2.4 Memory Bandwidth and Power

HBM3 memory power is a significant fraction of total GPU power:
- HBM3 memory consumes approximately 15-20% of total GPU power at full bandwidth
- Memory-intensive LLM inference workloads can have high memory utilization with moderate GPU compute utilization
- The ratio of GPU utilization to memory utilization is a useful feature for power prediction (as noted in PI-DLinear's alpha parameter)

### 2.5 Temperature-Power Coupling

The tight coupling between temperature and power operates through multiple mechanisms:
1. Higher temperature increases leakage current (exponential relationship)
2. Higher leakage increases power, which increases temperature (positive feedback loop)
3. Thermal throttling reduces clock speed to prevent overheating, reducing power
4. Cooling system effectiveness varies with ambient conditions

**For ENERGIVANU:** This coupling is exactly what PI-DLinear exploits. The thermal RC network captures these dynamics in a physically consistent way.

### 2.6 Cluster-Level Power Characteristics

At the 100K GPU scale:
- **Total compute power:** approximately 70 MW (100,000 x 700W)
- **Cooling overhead (PUE 1.2):** approximately 84 MW total facility power
- **Power variability:** Workload scheduling, batch boundaries, and job completions cause power swings of 5-15% over minutes
- **Ramp rates:** Power can change by several MW within minutes as large training jobs start/stop

---

## 3. xAI Colossus Infrastructure

### 3.1 Overview

| Attribute | Detail |
|-----------|--------|
| Operator | xAI (Elon Musk) |
| Location | Memphis, Tennessee (former Electrolux facility) |
| GPUs | 100,000+ NVIDIA H100 SXM5 |
| Build Time | approximately 122 days (2024) |
| Primary Use | Training Grok LLMs |
| Estimated Power | approximately 150+ MW total facility |

### 3.2 Power Infrastructure

**GPU Power Draw:**
- 100,000 H100 SXM5 at 700W TDP = 70 MW compute power alone
- With NVLink, memory, CPUs, networking: approximately 100 MW for compute infrastructure
- Cooling and facility overhead (PUE approximately 1.3-1.5): approximately 130-150 MW total

**Tesla Megapack Deployment:**
- Tesla Megapack battery units deployed at the site for power buffering
- Each Megapack: approximately 3.9 MWh capacity, approximately 1.9 MW output
- Purpose: buffer against grid instability, provide burst power during demand spikes
- Chemistry: LFP (Lithium Iron Phosphate) for safety and longevity
- Round-trip efficiency: approximately 90%+
- Response time: milliseconds (for frequency regulation)

**Grid Connection:**
- Memphis Light, Gas and Water (MLGW) supplies power
- Tennessee Valley Authority (TVA) involved in power supply
- Mobile natural gas generators also deployed on-site as backup
- Community concerns about grid strain and environmental impact

### 3.3 Expansion Plans

- Phase 2 planned: expansion to 200,000-300,000 NVIDIA B200 GPUs
- B200 TDP: approximately 1,000W (vs 700W for H100)
- Phase 2 power requirements: potentially 200-300 MW

### 3.4 Why 10-Minute Forecasting Matters

For Tesla Megapack battery signaling:
- **Charge/discharge optimization:** Predict power demand 10 min ahead to pre-charge batteries before demand spikes, or pre-discharge before demand drops
- **Grid stability:** Sudden 5-15% power swings (several MW) on a 100K GPU cluster can destabilize the local grid. 10-minute lookahead enables smooth ramping
- **Cost optimization:** Time-of-use electricity pricing; pre-charge from grid during cheap periods
- **Thermal management:** Batteries have optimal charge/discharge temperature ranges; predict-ahead enables thermal preconditioning

### 3.5 Scale of Power Forecasting Challenge

- **100,000 GPUs** with heterogeneous workloads (training, inference, mixed)
- **Power swings:** Job completions, new job starts, checkpoint saves, all-hands-reduce synchronization events
- **Forecasting target:** Aggregate power of the entire cluster, 10 minutes ahead, at 1-minute resolution
- **Accuracy requirement:** For battery signaling, approximately 1-2% RMSE on aggregate power is desirable

---

## 4. NVIDIA DCGM: Data Center GPU Manager

### 4.1 Overview

DCGM (Data Center GPU Manager) is NVIDIA's official toolset for managing and monitoring NVIDIA GPUs in data center environments. It provides diagnostics, health monitoring, policy management, and telemetry collection.

### 4.2 Key Metrics for Power Forecasting

#### Critical Field IDs:

| Field ID | Constant Name | Description | Unit |
|----------|---------------|-------------|------|
| 152 | `DCGM_FI_DEV_POWER_USAGE` | Current power draw | Watts |
| 153 | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | Cumulative energy | mJ |
| 154 | `DCGM_FI_DEV_POWER_MGMT_LIMIT` | Power management cap | W |
| 151 | `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature | deg C |
| 150 | `DCGM_FI_DEV_MEMORY_TEMP` | HBM temperature | deg C |
| 203 | `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilization | % |
| 204 | `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory copy utilization | % |
| 100 | `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency | MHz |
| 101 | `DCGM_FI_DEV_MEM_CLOCK` | Memory clock frequency | MHz |
| 252 | `DCGM_FI_DEV_FB_FREE` | Framebuffer free | MiB |
| 253 | `DCGM_FI_DEV_FB_USED` | Framebuffer used | MiB |
| 200 | `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe TX throughput | KB/s |
| 201 | `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` | PCIe RX throughput | KB/s |

#### Additional Useful Metrics:
- `DCGM_FI_DEV_SM_ACTIVE` (101): SM activity percentage
- `DCGM_FI_DEV_TENSOR_ACTIVE` (111): Tensor core activity
- `DCGM_FI_DEV_DRAM_ACTIVE` (112): DRAM activity
- `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` (156/157): NVLink traffic
- `DCGM_FI_DEV_XID_ERRORS` (240): GPU error events
- ECC errors (various): Memory error counts

### 4.3 Collection Methods

#### Method 1: dcgm-exporter (Prometheus)

The standard production approach. Runs as a daemon, exports metrics in Prometheus format.

```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  template:
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
        ports:
        - containerPort: 9400
          name: metrics
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]  # or use privileged mode
```

**Custom counters file** (default-counters.csv):
```csv
DCGM_FI_DEV_SM_CLOCK,       clock_sm,           SM clock frequency (in MHz).
DCGM_FI_DEV_MEM_CLOCK,      clock_mem,          Memory clock frequency (in MHz).
DCGM_FI_DEV_MEMORY_TEMP,    temp_memory,        Memory temperature (in C).
DCGM_FI_DEV_GPU_TEMP,       temp_gpu,           GPU temperature (in C).
DCGM_FI_DEV_POWER_USAGE,    power_usage,        Power draw (in W).
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, energy_consumption, Total energy consumption (in mJ).
DCGM_FI_DEV_GPU_UTIL,       gpu_util,           GPU utilization (in %).
DCGM_FI_DEV_MEM_COPY_UTIL,  mem_copy_util,      Memory utilization (in %).
DCGM_FI_DEV_FB_FREE,        fb_free,            Framebuffer memory free (in MiB).
DCGM_FI_DEV_FB_USED,        fb_used,            Framebuffer memory used (in MiB).
```

**Prometheus query examples:**
```promql
# Aggregate power across all GPUs
sum(DCGM_FI_DEV_POWER_USAGE)

# Average GPU utilization
avg(DCGM_FI_DEV_GPU_UTIL)

# Power per node
sum by (instance) (DCGM_FI_DEV_POWER_USAGE)

# Rate of energy consumption (watts from cumulative)
rate(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION[1m])
```

#### Method 2: DCGM C API (libdcgm.so)

For custom collection pipelines:

```c
#include <dcgm_agent.h>

// Initialize
dcgmInit();
dcgmConnect("localhost", &dcgmHandle);

// Create field group with power-relevant fields
dcgmFieldGrp_t fieldGroup;
unsigned int fieldIds[] = {152, 151, 150, 203, 204, 100, 101};
dcgmFieldGroupCreate(dcgmHandle, 7, fieldIds, "power_fields", &fieldGroup);

// Create watch (1-second update, 5-minute retention)
dcgmWatchFields(dcgmHandle, groupId, gpuId, 1000000, 300.0, 7);

// Fetch latest values
dcgmFieldValue_v2 values[7];
dcgmGetLatestValues(dcgmHandle, gpuId, fieldGroup, values);
```

#### Method 3: nvidia-smi (CLI)

For quick prototyping:
```bash
# Query power and utilization every second
nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory \
           --format=csv -l 1

# Set power limit
nvidia-smi -pl 600  # Cap at 600W

# Query specific fields
nvidia-smi -q -d POWER,TEMPERATURE,UTILIZATION,CLOCK
```

#### Method 4: NVML (NVIDIA Management Library)

Low-level Python bindings:
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Power
power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)

# Temperature
gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
mem_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)

# Utilization
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
gpu_util = util.gpu      # 0-100
mem_util = util.memory   # 0-100

# Clocks
sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_ID_CURRENT)
mem_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_ID_CURRENT)

# Memory
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
fb_used = mem_info.used
fb_free = mem_info.free
```

### 4.4 Data Pipeline Architecture for ENERGIVANU

```
[100K GPUs] --> [DCGM agents per node] --> [dcgm-exporter per node]
                                                  |
                                          [Prometheus cluster]
                                                  |
                                          [TimescaleDB / InfluxDB]
                                                  |
                                          [Feature engineering]
                                                  |
                                          [PI-DLinear model]
                                                  |
                                          [Battery signal API]
                                                  |
                                          [Tesla Megapack controller]
```

**Recommended collection interval:** 1 second (DCGM can do sub-second), aggregated to 1 minute for model input (matching PI-DLinear's training granularity).

**Storage:** At 1 second, 100K GPUs, 7 fields = approximately 700K data points/second. TimescaleDB or InfluxDB can handle this with proper partitioning.

---

## 5. Power Throttling and DVFS Effects

### 5.1 DVFS Fundamentals

Dynamic Voltage and Frequency Scaling (DVFS) is the primary mechanism by which NVIDIA GPUs manage power consumption.

**Power-voltage-frequency relationship:**
```
P = C_eff * V^2 * f * alpha + P_static(V, T)
```

Since voltage scales with frequency (approximately V proportional to f in high-performance regimes):
```
P approximately proportional to f^3  (at constant activity)
```

This cubic relationship means the last 10% of clock speed may cost 20-30% more power.

### 5.2 H100 DVFS Operation

**Clock domains:**
- SM (Streaming Multiprocessor) clock: base 1095 MHz, boost up to 1980 MHz
- HBM3 memory clock: approximately 2619 MHz (fixed in most configurations)
- NVLink clock: separate domain

**DVFS table:** NVIDIA GPUs have a voltage-frequency lookup table. The governor selects the highest frequency state that fits within the power envelope.

**Power capping via nvidia-smi:**
```bash
nvidia-smi -pl 700   # Default (max performance)
nvidia-smi -pl 600   # Moderate cap (approximately 5-10% clock reduction)
nvidia-smi -pl 400   # Aggressive cap (approximately 15-25% clock reduction)
```

### 5.3 Throttling Mechanisms

#### Power Throttling (Power Cap Enforcement):
- When workload power exceeds the set power limit, the GPU reduces clock frequency
- This is the most common throttling scenario in production
- PI-DLinear defines this as "sudden power drops exceeding 15%"

#### Thermal Throttling:
- GPU die temperature exceeds T_jmax (83 deg C for H100)
- Clock reduced in steps to prevent damage
- In the PI-DLinear dataset, critical temperatures never reached 80-90 deg C limits
- Thermal throttling is rare in well-cooled data centers

#### Voltage Throttling:
- VRM (Voltage Regulator Module) thermal protection
- Reduces available voltage, forcing lower clocks

#### Software Throttling:
- nvidia-smi power limit enforcement
- Application-level power management
- Kubernetes resource limits

### 5.4 Impact on Power Forecasting

**Why throttling matters for ENERGIVANU:**
1. **Prediction accuracy:** A model that ignores throttling will overpredict power during high-utilization periods
2. **Battery signaling:** Unexpected throttling events cause power drops that the battery must absorb. Predicting these events allows preemptive battery discharge
3. **Physical consistency:** PI-DLinear's throttling loss (`L_throttle`) explicitly handles this

**Observable signatures of throttling:**
- GPU utilization stays high (>90%) but power drops
- Temperature at or near limits
- Clock frequency decreases
- Power oscillations near the power cap (rapid throttle/unthrottle cycles)

### 5.5 Data Center Power Capping Strategies

**Static power capping:** Set a fixed power limit per GPU (e.g., 600W instead of 700W). Reduces peak power by approximately 14% with approximately 5-10% performance loss. Common for cost optimization.

**Dynamic power capping:** Adjust power limits based on:
- Grid power availability
- Electricity pricing (time-of-use)
- Cooling capacity
- Battery state of charge

**For ENERGIVANU:** The 10-minute forecast can inform dynamic power capping. If a demand spike is predicted, pre-position the battery and optionally reduce GPU power caps to flatten the spike.

---

## 6. Related Research Papers

### 6.1 Time Series Forecasting Models (Baselines from PI-DLinear)

| Model | Type | Key Innovation |
|-------|------|----------------|
| **Informer** (Zhou et al., 2021) | Transformer | ProbSparse attention, long sequence forecasting (AAAI 2021 Best Paper) |
| **Autoformer** (Wu et al., 2021) | Transformer | Auto-correlation mechanism with series decomposition |
| **FEDformer** (Zhou et al., 2022) | Transformer | Frequency-enhanced decomposition |
| **PatchTST** (Nie et al., 2023) | Transformer | Patching + channel independence; "A Time Series is Worth 64 Words" |
| **iTransformer** | Transformer | Inverted attention on variate dimension |
| **DLinear** (Zeng et al., 2023) | Linear | Decomposition + linear; surprisingly beats transformers |
| **NLinear** (Zeng et al., 2023) | Linear | Normalization + linear layer |
| **TiDE** | MLP | Dense encoder-decoder for time series |
| **FiLM** | MLP | Frequency improved Legendre Memory |
| **TimesNet** | CNN | Temporal 2D variation modeling |
| **Pyraformer** | Transformer | Pyramidal attention |
| **Crossformer** | Transformer | Cross-dimension attention |
| **LightTS** | MLP | Lightweight time series model |
| **TSMixer** | MLP | Time series mixer architecture |
| **Reformer** | Transformer | LSH attention for efficiency |
| **Nonstationary Transformer** | Transformer | De-stationary attention |

### 6.2 Data Center Power Forecasting Research

**Google DeepMind Data Center Cooling (2016):**
- Used deep neural networks to reduce data center cooling energy by 40%
- Trained on historical sensor data (temperatures, power, pump speeds)
- Applied recommendations via automated control system
- Demonstrated feasibility of ML-based data center energy optimization

**MIT Supercloud Dataset (Samsi et al., 2021):**
- The dataset used in PI-DLinear
- 238 days of GPU telemetry from a 448-GPU V100 cluster
- Includes job-level metadata for workload characterization
- Publicly available for research

**Physics-Informed Neural Networks (PINNs) for Data Centers:**
- Combine physics equations (energy balance, heat transfer) with deep learning
- Embed conservation laws directly in the loss function
- Benefits: better generalization, physical plausibility, less training data needed
- Applications: thermal modeling, cooling optimization, power prediction

**Google's 4D Data Center Digital Twins:**
- Real-time simulation of data center thermal dynamics
- Physics-based models coupled with sensor data
- Used for cooling optimization and capacity planning

### 6.3 GPU Power Modeling Research

**Empirical GPU Power Models:**
- Linear regression on utilization, clock, temperature features
- R-squared typically 0.85-0.95 for single workloads
- Degrades significantly for heterogeneous workloads (the real-world case)

**Workload-Aware Power Models:**
- Different power profiles for compute-bound vs memory-bound workloads
- Require workload classification as a preprocessing step
- More accurate but harder to maintain

**NVIDIA's Internal Power Model:**
- Built into the GPU's power management firmware
- Estimates power from clock frequency, voltage, and activity counters
- Not directly accessible but manifests as the power reading from nvidia-smi

### 6.4 Battery-Energy Storage for Data Centers

**Grid-Scale Battery Applications:**
- Peak shaving: reduce grid power draw during demand peaks
- Frequency regulation: provide fast-response grid stabilization
- Arbitrage: charge during cheap electricity periods, discharge during expensive periods
- Backup: uninterruptible power during grid outages

**Tesla Megapack for Data Centers:**
- Response time: milliseconds (vs seconds for diesel generators)
- Cycle efficiency: approximately 90%
- Cycle life: approximately 10,000+ cycles (LFP chemistry)
- Scalable from MWh to GWh

---

## 7. Implementation Architecture for ENERGIVANU

### 7.1 System Architecture

```
                    +-----------------------+
                    |   xAI Colossus        |
                    |   100K H100 GPUs      |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   DCGM Agents         |
                    |   (per GPU node)      |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   dcgm-exporter       |
                    |   (Prometheus format) |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Time-Series DB      |
                    |   (TimescaleDB)       |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Feature Pipeline    |
                    |   (1-min aggregation) |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   PI-DLinear Model    |
                    |   (PyTorch)           |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Signal Generator    |
                    |   (charge/discharge)  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Tesla Megapack      |
                    |   Controller API      |
                    +-----------------------+
```

### 7.2 Model Implementation (PyTorch Pseudocode)

```python
import torch
import torch.nn as nn

class DLinear(nn.Module):
    """Base DLinear model for time series decomposition + linear prediction."""
    def __init__(self, seq_len, pred_len, num_channels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Separate linear layers for trend and seasonal
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        # Moving average for trend extraction
        self.ma_kernel_size = 25  # Typical value

    def forward(self, x):
        # x: (batch, seq_len, channels)
        # Extract trend via moving average
        trend = self._moving_average(x)
        seasonal = x - trend

        # Apply linear layers per channel
        # Reshape for channel-independent linear
        B, L, C = x.shape
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))  # (B, C, T)
        trend_out = self.linear_trend(trend.permute(0, 2, 1))  # (B, C, T)

        output = seasonal_out + trend_out
        return output.permute(0, 2, 1)  # (B, T, C)

    def _moving_average(self, x):
        kernel = torch.ones(1, 1, self.ma_kernel_size, device=x.device) / self.ma_kernel_size
        # Pad and convolve per channel
        B, L, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, 1, L)
        padded = nn.functional.pad(x_flat, (self.ma_kernel_size // 2, self.ma_kernel_size // 2), mode='replicate')
        trend = nn.functional.conv1d(padded, kernel).reshape(B, C, L)
        return trend.permute(0, 2, 1)


class ThermalRCNetwork:
    """Physics model for GPU thermal dynamics."""
    def __init__(self, C_g, C_m, R_ga, R_ma, R_gm, T_a, alpha):
        self.C_g = C_g      # GPU thermal capacitance (J/K)
        self.C_m = C_m      # Memory thermal capacitance (J/K)
        self.R_ga = R_ga    # GPU-to-ambient resistance (K/W)
        self.R_ma = R_ma    # Memory-to-ambient resistance (K/W)
        self.R_gm = R_gm    # GPU-to-memory resistance (K/W)
        self.T_a = T_a      # Ambient temperature (K)
        self.alpha = alpha  # Power split fraction

    def compute_residual(self, P_pred, T_g, T_m, dT_g_dt, dT_m_dt, d2T_g_dt2):
        """
        Compute physics residual: how well predictions satisfy the thermal ODE.

        dP/dt = (1/alpha) * [C_g * d2Tg/dt2 + (1/R_ga)*dTg/dt + (1/R_gm)*(dTg/dt - dTm/dt)]
        """
        dP_dt_pred = (1.0 / self.alpha) * (
            self.C_g * d2T_g_dt2
            + (1.0 / self.R_ga) * dT_g_dt
            + (1.0 / self.R_gm) * (dT_g_dt - dT_m_dt)
        )
        return dP_dt_pred


class PILoss(nn.Module):
    """Physics-Informed loss for GPU power forecasting."""
    def __init__(self, thermal_model, lambda_u=1.0, lambda_r=0.005, lambda_theta=0.005):
        super().__init__()
        self.thermal = thermal_model
        self.lambda_u = lambda_u
        self.lambda_r = lambda_r
        self.lambda_theta = lambda_theta
        # Self-adaptive weights (log-space)
        self.eta_u = nn.Parameter(torch.log(torch.tensor(lambda_u)))
        self.eta_r = nn.Parameter(torch.log(torch.tensor(lambda_r)))
        self.eta_theta = nn.Parameter(torch.log(torch.tensor(lambda_theta)))

    def forward(self, P_pred, P_true, T_g, T_m, util, theta_U=0.9):
        # Data loss (MSE)
        L_data = torch.mean((P_pred - P_true) ** 2)

        # Physics residual loss (requires autograd for dP/dt)
        L_physics = self._physics_residual(P_pred, T_g, T_m)

        # Throttling constraint loss
        L_throttle = self._throttle_constraint(P_pred, util, T_g, theta_U)

        # Self-adaptive weighting
        lambda_u = torch.exp(self.eta_u).clamp(1e-5, 1e5)
        lambda_r = torch.exp(self.eta_r).clamp(1e-5, 1e5)
        lambda_theta = torch.exp(self.eta_theta).clamp(1e-5, 1e5)

        total_loss = lambda_u * L_data + lambda_r * L_physics + lambda_theta * L_throttle
        return total_loss

    def _physics_residual(self, P_pred, T_g, T_m):
        """Compute ODE residual using automatic differentiation."""
        # Compute time derivatives of temperature via autograd
        # (Simplified -- actual implementation requires careful autograd setup)
        dT_g = torch.gradient(T_g, dim=1)[0]
        dT_m = torch.gradient(T_m, dim=1)[0]
        d2T_g = torch.gradient(dT_g, dim=1)[0]
        dP_dt = torch.gradient(P_pred, dim=1)[0]

        # Expected dP/dt from thermal model
        dP_dt_expected = self.thermal.compute_residual(
            P_pred, T_g, T_m, dT_g, dT_m, d2T_g
        )

        return torch.mean((dP_dt - dP_dt_expected) ** 2)

    def _throttle_constraint(self, P_pred, util, T_g, theta_U, theta_T_pct=0.95):
        """Penalize power increases at high utilization."""
        dP = P_pred[:, 1:] - P_pred[:, :-1]  # Power change
        u = util[:, 1:]  # Utilization at corresponding timesteps

        # High utilization mask
        high_util_mask = (u > theta_U).float()

        # Penalize positive power changes when utilization is high
        L_high = torch.mean(high_util_mask * torch.relu(dP) ** 2)

        # Extra penalty when also at high temperature
        theta_T = torch.quantile(T_g, theta_T_pct)
        high_temp_mask = (T_g[:, 1:] > theta_T).float()
        stress_mask = high_util_mask * high_temp_mask
        L_stress = torch.mean(stress_mask * torch.relu(dP) ** 2)

        return L_high + L_stress


class PIDLinear(nn.Module):
    """Physics-Informed DLinear for GPU power forecasting."""
    def __init__(self, seq_len, pred_len, num_channels=5):
        super().__init__()
        self.dlinear = DLinear(seq_len, pred_len, num_channels)
        # Initialize thermal model with default (V100) parameters
        # MUST be re-estimated for H100
        self.thermal = ThermalRCNetwork(
            C_g=5.408e6, C_m=5.481e6,
            R_ga=2.037e-3, R_ma=2.055e-3, R_gm=6.064e-4,
            T_a=300.15,  # 27 deg C in Kelvin
            alpha=0.5085
        )
        self.pi_loss = PILoss(self.thermal)

    def forward(self, x):
        return self.dlinear(x)

    def compute_loss(self, P_pred, P_true, T_g, T_m, util):
        return self.pi_loss(P_pred, P_true, T_g, T_m, util)
```

### 7.3 RC Parameter Estimation for H100

The thermal RC parameters from the paper were fitted to V100 GPUs. For H100, you must re-estimate:

```python
import numpy as np
from scipy.optimize import minimize

def estimate_rc_parameters(P_measured, T_g_measured, T_m_measured, dt=60.0):
    """
    Estimate thermal RC parameters using Recursive Least Squares.

    Args:
        P_measured: Power measurements (W), shape (N,)
        T_g_measured: GPU temperature (deg C), shape (N,)
        T_m_measured: Memory temperature (deg C), shape (N,)
        dt: Sampling interval in seconds (default 60s for 1-min data)

    Returns:
        dict with C_g, C_m, R_ga, R_ma, R_gm, alpha, T_a
    """
    T_a = np.min(T_g_measured)  # Approximate ambient

    def residual(params):
        C_g, C_m, R_ga, R_ma, R_gm, alpha = params
        T_g = T_g_measured
        T_m = T_m_measured
        P = P_measured

        # Compute temperature derivatives (finite differences)
        dT_g = np.gradient(T_g, dt)
        dT_m = np.gradient(T_m, dt)

        # GPU node: C_g * dTg/dt = alpha*P - (Tg-Ta)/Rga - (Tg-Tm)/Rgm
        res_g = C_g * dT_g - alpha * P + (T_g - T_a) / R_ga + (T_g - T_m) / R_gm

        # Memory node: C_m * dTm/dt = (1-alpha)*P - (Tm-Ta)/Rma + (Tg-Tm)/Rgm
        res_m = C_m * dT_m - (1 - alpha) * P + (T_m - T_a) / R_ma - (T_g - T_m) / R_gm

        return np.sum(res_g ** 2) + np.sum(res_m ** 2)

    # Initial guess (scaled from V100 parameters)
    x0 = [5.408e6, 5.481e6, 2.037e-3, 2.055e-3, 6.064e-4, 0.5]

    # All parameters must be positive; alpha in [0, 1]
    bounds = [
        (1e5, 1e8),    # C_g
        (1e5, 1e8),    # C_m
        (1e-5, 1e-1),  # R_ga
        (1e-5, 1e-1),  # R_ma
        (1e-5, 1e-1),  # R_gm
        (0.0, 1.0),    # alpha
    ]

    result = minimize(residual, x0, method='L-BFGS-B', bounds=bounds)

    return {
        'C_g': result.x[0],
        'C_m': result.x[1],
        'R_ga': result.x[2],
        'R_ma': result.x[3],
        'R_gm': result.x[4],
        'alpha': result.x[5],
        'T_a': T_a
    }
```

### 7.4 Data Collection Pipeline

```python
# Using NVML for direct GPU telemetry collection
import pynvml
import time
import json

class GPUTelemetryCollector:
    def __init__(self, gpu_ids=None):
        pynvml.nvmlInit()
        if gpu_ids is None:
            gpu_ids = list(range(pynvml.nvmlDeviceGetCount()))
        self.gpu_ids = gpu_ids
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids]

    def collect_sample(self):
        """Collect one telemetry sample from all GPUs."""
        timestamp = time.time()
        samples = []

        for gpu_id, handle in zip(self.gpu_ids, self.handles):
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

            samples.append({
                'timestamp': timestamp,
                'gpu_id': gpu_id,
                'power_w': power,
                'gpu_temp_c': gpu_temp,
                'mem_temp_c': mem_temp,
                'gpu_util_pct': util.gpu,
                'mem_util_pct': util.memory,
                'sm_clock_mhz': clocks,
                'mem_clock_mhz': mem_clocks,
            })

        return samples

    def collect_loop(self, interval_sec=1.0, callback=None):
        """Continuous collection loop."""
        while True:
            samples = self.collect_sample()
            if callback:
                callback(samples)
            time.sleep(interval_sec)
```

### 7.5 Battery Signal Generation

```python
class BatterySignalGenerator:
    """Generate charge/discharge signals for Tesla Megapack based on power forecast."""

    def __init__(self, battery_capacity_mwh, max_power_mw, soc_min=0.1, soc_max=0.9):
        self.capacity = battery_capacity_mwh
        self.max_power = max_power_mw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.current_soc = 0.5  # State of charge (0-1)

    def generate_signal(self, current_power_mw, predicted_power_mw, horizon_min=10):
        """
        Generate battery command based on current and predicted power.

        Args:
            current_power_mw: Current aggregate GPU power draw (MW)
            predicted_power_mw: Predicted power 10 minutes ahead (MW)
            horizon_min: Prediction horizon in minutes

        Returns:
            dict with battery command
        """
        delta_power = predicted_power_mw - current_power_mw
        ramp_rate = delta_power / horizon_min  # MW per minute

        if delta_power > 0:
            # Demand increasing -- pre-charge battery now, discharge later
            # Charge at current rate to prepare for future discharge
            charge_power = min(delta_power * 0.5, self.max_power, 
                             (self.soc_max - self.current_soc) * self.capacity / (horizon_min / 60))
            return {
                'action': 'charge',
                'power_mw': charge_power,
                'reason': f'Demand spike predicted: +{delta_power:.1f} MW in {horizon_min} min',
                'confidence': 'high' if abs(delta_power) > 5 else 'medium'
            }
        elif delta_power < 0:
            # Demand decreasing -- discharge battery now to smooth the drop
            discharge_power = min(abs(delta_power) * 0.5, self.max_power,
                                (self.current_soc - self.soc_min) * self.capacity / (horizon_min / 60))
            return {
                'action': 'discharge',
                'power_mw': discharge_power,
                'reason': f'Demand drop predicted: {delta_power:.1f} MW in {horizon_min} min',
                'confidence': 'high' if abs(delta_power) > 5 else 'medium'
            }
        else:
            return {
                'action': 'idle',
                'power_mw': 0,
                'reason': 'No significant change predicted',
                'confidence': 'high'
            }
```

### 7.6 Key Implementation Considerations

1. **Model retraining schedule:** Retrain weekly or when workload composition changes significantly. The physics parameters (RC network) are hardware-specific and stable, but the DLinear weights adapt to workload patterns.

2. **Aggregate vs per-GPU forecasting:** For 100K GPUs, forecast aggregate power. Per-GPU forecasting is unnecessary and computationally expensive. The aggregate is smoother and more predictable.

3. **Feature aggregation:** Average GPU utilization, max temperature, total power. The thermal dynamics at aggregate level follow the same ODEs with scaled parameters.

4. **Handling missing data:** DCGM may have gaps. Use forward-fill for short gaps (< 5 min), model re-initialization for longer gaps.

5. **Latency requirements:** Model inference is essentially instant (single linear layer). Data collection latency (DCGM to model) should be under 5 seconds. Battery command latency should be under 10 seconds.

6. **Model monitoring:** Track prediction accuracy in real-time. Retrain if RMSE exceeds 2% on a rolling 1-hour window.

7. **Cold start:** On initial deployment without H100-specific RC parameters, use the V100 parameters from the paper as a starting point and fine-tune with RLS as data accumulates (converges in approximately 1-2 days of operation).

---

## References

1. AlShaikh Saleh, M., Chawla, S., Bayhan, S., Abu-Rub, H., Ghrayeb, A. (2025). "A Physics-Aware Framework for Short-Term GPU Power Forecasting of AI Data Centers." arXiv:2605.04074.

2. Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. (DLinear paper)

3. Samsi, S., et al. (2021). "From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference." MIT Supercloud Dataset.

4. Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021.

5. Wu, H., et al. (2021). "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting." NeurIPS 2021.

6. Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.

7. NVIDIA. "DCGM: Data Center GPU Manager." https://github.com/NVIDIA/DCGM

8. NVIDIA. "dcgm-exporter: Prometheus exporter for DCGM metrics." https://github.com/NVIDIA/dcgm-exporter

9. NVIDIA. "H100 Tensor Core GPU Datasheet." https://www.nvidia.com/en-us/data-center/h100/

10. Google DeepMind. "DeepMind AI Reduces Google Data Centre Cooling Bill by 40%." 2016.

11. Tesla. "Megapack." https://tesla.com/megapack

---

*Document generated for ENERGIVANU project. All implementation code is pseudocode intended to guide development -- not production-ready. Test thoroughly before deployment.*
