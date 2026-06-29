# Energivanu Dataset Documentation

This document describes the data formats, features, expected ranges,
and collection methods used across the Energivanu project.

---

## 1. The 15 Features

The PEB (Power-Efficiency-Battery) model consumes **15 features** per timestep.
They are derived from raw GPU telemetry via `src/energivanu/data.py`.

| #  | Column Name            | Unit / Type    | Expected Range       | Description                                              |
|----|------------------------|----------------|----------------------|----------------------------------------------------------|
| 1  | `facility_mw`          | MW             | 0 – 500              | Total facility power draw, scaled from single-node data  |
| 2  | `power_roc`            | MW/s           | -500 – 500           | First derivative (rate of change) of facility power      |
| 3  | `power_roc2`           | MW/s²          | -200 – 200           | Second derivative (acceleration) of facility power       |
| 4  | `power_roll_mean`      | MW             | 0 – 500              | Rolling mean of facility power (window = 250 samples)    |
| 5  | `power_roll_std`       | MW             | 0 – 100              | Rolling standard deviation of facility power             |
| 6  | `gpu_avg_power_norm`   | Normalised     | 0 – 1.0              | Average GPU power / 700 W (H100 TDP)                     |
| 7  | `gpu_max_power_norm`   | Normalised     | 0 – 1.0              | Maximum GPU power across 8 GPUs / 700 W                  |
| 8  | `gpu_avg_temp_norm`    | Normalised     | 0 – 1.1              | Average GPU temperature / 100 °C                         |
| 9  | `gpu_max_temp_norm`    | Normalised     | 0 – 1.1              | Maximum GPU temperature / 100 °C                         |
| 10 | `gpu_avg_util_norm`    | Normalised     | 0 – 1.0              | Average GPU utilisation / 100 %                          |
| 11 | `gpu_avg_mem_util_norm`| Normalised     | 0 – 1.0              | Average GPU memory utilisation / 100 %                   |
| 12 | `cpu_util_norm`        | Normalised     | 0 – 1.0              | CPU utilisation / 100 %                                  |
| 13 | `hour_sin`             | Cyclical       | -1.0 – 1.0           | sin(2π × hour / 24) — encodes time of day                |
| 14 | `hour_cos`             | Cyclical       | -1.0 – 1.0           | cos(2π × hour / 24) — encodes time of day                |
| 15 | `is_allreduce`         | Binary         | 0 or 1               | 1 if GPU util > 80 % AND memory util < 30 % (all-reduce) |

### Feature Normalisation Reference

| Raw Feature            | Divisor   | Rationale                        |
|------------------------|-----------|----------------------------------|
| GPU power (W)          | 700       | H100 TDP ≈ 700 W                |
| GPU temperature (°C)   | 100       | Max safe operating temp ≈ 100 °C |
| GPU utilisation (%)    | 100       | Already a percentage             |
| GPU memory util (%)    | 100       | Already a percentage             |
| CPU utilisation (%)    | 100       | Already a percentage             |

---

## 2. Data Sources

### 2.1 nvidia-smi (Own Collection)

**Tool:** `nvidia-smi --query-gpu=... --format=csv -l 1`

**How to collect:**
```bash
# Single snapshot
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits

# Continuous collection (1-second interval)
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1 > telemetry_raw.csv
```

**Raw CSV format (per-GPU rows):**
```csv
2026/06/28 12:00:00.000, 0, 65.23, 42, 3, 1.2
2026/06/28 12:00:00.000, 1, 67.10, 43, 5, 1.5
2026/06/28 12:00:00.000, 2, 64.80, 41, 2, 1.1
...
```

**Column mapping (raw → processed):**
| Raw nvidia-smi Column   | Processed Feature        |
|-------------------------|--------------------------|
| `power.draw` (per GPU)  | `gpu_avg_power_norm`     |
| `temperature.gpu`       | `gpu_avg_temp_norm`      |
| `utilization.gpu`       | `gpu_avg_util_norm`      |
| `utilization.memory`    | `gpu_avg_mem_util_norm`  |

**License:** Self-collected — **zero restrictions**, fully commercial.

### 2.2 Alibaba GPU Trace 2020

**Source:** [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata)

**Dataset:** Cluster trace GPU v2020 — 6,500 GPUs, 2 months of production data

**Key tables:**
- `pai_sensor_table` — GPU sensor readings (utilisation, memory, temperature)
- `pai_instance_table` — Per-instance resource metrics
- `pai_machine_spec` — Hardware specifications

**Feature mapping (Alibaba → our 15):**
| Alibaba Column          | Our Feature              | Transformation              |
|-------------------------|--------------------------|-----------------------------|
| `gpu_util`              | `gpu_avg_util_norm`      | / 100                       |
| `gpu_memory_util`       | `gpu_avg_mem_util_norm`  | / 100                       |
| `gpu_temp`              | `gpu_avg_temp_norm`      | / 100                       |
| *(derived)*             | `gpu_avg_power_norm`     | power model: util × 630 + 70|
| *(derived)*             | `facility_mw`            | scale to 200k GPUs          |

**Citation (required):**
```bibtex
@inproceedings{weng2022mlaas,
  title={{MLaaS} in the Wild: Workload Analysis and Scheduling
         in Large-Scale Heterogeneous {GPU} Clusters},
  author={Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and
          Wang, Wei and Wang, Cheng and He, Jian and Li, Yong
          and Zhang, Liping and Lin, Wei and Ding, Yu},
  booktitle={NSDI '22},
  year={2022}
}
```

**License:** CC BY 4.0 — **fully commercial-safe** with attribution.

### 2.3 York University H100 Dataset

**Source:** FigShare (High-resolution AI Data Center Training Workloads)

**Content:** 8 × H100 GPU node, 20 ms resolution, LLM training workloads

**Raw column names:**
```
timestamp, gpu0_power_W, gpu0_temp_C, gpu0_utilization_percent, gpu0_mem_utilization,
gpu1_power_W, ..., cpu_utilization_percent
```

**Feature mapping (York → our 15):**
| York Column                  | Our Feature              |
|------------------------------|--------------------------|
| `gpu{i}_power_W` (avg 0–7)  | `gpu_avg_power_norm`     |
| `gpu{i}_temp_C` (avg 0–7)   | `gpu_avg_temp_norm`      |
| `gpu{i}_utilization_percent` | `gpu_avg_util_norm`      |
| `gpu{i}_mem_utilization`     | `gpu_avg_mem_util_norm`  |
| `cpu_utilization_percent`    | `cpu_util_norm`          |

**License:** CC BY-NC-ND 4.0 — **research only, NOT for commercial use.**

⚠️ **WARNING:** York data must NEVER appear in commercial training pipelines.
Use `scripts/check_compliance.py` to verify.

---

## 3. Processed Training Format

After processing through `src/energivanu/data.py`, training data looks like:

**Example CSV (processed, 15 features):**
```csv
facility_mw,power_roc,power_roc2,power_roll_mean,power_roll_std,gpu_avg_power_norm,gpu_max_power_norm,gpu_avg_temp_norm,gpu_max_temp_norm,gpu_avg_util_norm,gpu_avg_mem_util_norm,cpu_util_norm,hour_sin,hour_cos,is_allreduce
182.345,0.523,0.012,181.200,3.450,0.892,0.934,0.580,0.610,0.720,0.450,0.650,0.2588,0.9659,0
183.120,0.775,0.252,181.500,3.520,0.901,0.945,0.585,0.615,0.735,0.460,0.660,0.2588,0.9659,0
182.890,-0.230,-1.005,181.800,3.480,0.898,0.940,0.583,0.612,0.725,0.455,0.655,0.2588,0.9659,0
184.500,1.610,1.840,182.100,3.600,0.915,0.960,0.590,0.620,0.750,0.470,0.670,0.2588,0.9659,1
185.200,0.700,-0.910,182.500,3.650,0.922,0.968,0.595,0.625,0.760,0.475,0.675,0.2588,0.9659,0
```

**Sequence format (NumPy arrays for training):**
- `X`: shape `(N, 30, 15)` — 30 timesteps × 15 features
- `Y_power`: shape `(N, 10)` — 10-step power prediction target (MW)
- `Y_signal`: shape `(N,)` — classification label (0=hold, 1=discharge, 2=charge)

---

## 4. Data Validation

Use the built-in validator to check data quality before training:

```python
from energivanu.data.validator import DataQualityValidator

validator = DataQualityValidator()
report = validator.validate("data/telemetry.csv")

# Check results
print(f"Valid: {report.is_valid}")
print(f"NaN columns: {list(report.nan_summary.keys())}")
print(f"Outlier columns: {list(report.outlier_summary.keys())}")
print(f"Time-series gaps: {report.gap_summary.get('gap_count', 0)}")

# Access per-column statistics
for col, s in report.stats.items():
    print(f"{col}: mean={s['mean']:.4f}, std={s['std']:.4f}")
```

---

## 5. File Structure

```
data/
├── README.md              ← This file
├── telemetry.db           ← SQLite telemetry store (from nvidia-smi collector)
├── telemetry.csv          ← CSV telemetry export
├── real_h100/             ← York University H100 data (RESEARCH ONLY)
│   └── Node_Dataset/
│       └── Text Generation LLMs/
│           └── H100/
│               └── *.csv
└── alibaba_gpu_trace/     ← Alibaba GPU Trace 2020 (CC BY 4.0)
    └── pai_sensor_table.csv
```

---

## 6. License Summary

| Source         | License            | Commercial Use | Attribution Required |
|----------------|--------------------|----------------|----------------------|
| Self-collected | Public Domain      | ✅ Yes          | No                   |
| Alibaba 2020   | CC BY 4.0         | ✅ Yes          | Yes (cite NSDI '22)  |
| York H100      | CC BY-NC-ND 4.0   | ❌ No           | Yes                  |
| MIT Supercloud | CC BY-NC-ND 4.0   | ❌ No           | Yes                  |

**Rule:** Only Alibaba + self-collected data may enter the commercial training pipeline.
Run `scripts/check_compliance.py` to verify no NC-licensed data is present.
