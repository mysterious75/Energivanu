# Data Pipeline — Alibaba GPU Trace 2020

## Data Source
- **Dataset:** Alibaba GPU Trace 2020 (cluster-trace-gpu-v2020)
- **Source:** https://github.com/alibaba/clusterdata
- **License:** CC BY 4.0 (commercial use allowed)
- **Paper:** "Characterizing and Profiling GPU Workloads on Alibaba" (NSDI '22)
- **Machines:** 6,500 GPUs (P100, T4, V100)

## Raw Files Downloaded

| File | Size | Rows | Description |
|------|------|------|-------------|
| `pai_sensor_table.csv` | 1,057 MB | 30,33,232 | Per-GPU utilization (cpu_usage, gpu_wrk_util, avg_mem, etc.) |
| `pai_machine_metric.csv` | 437 MB | 20,09,423 | Machine-level metrics (cpu, gpu, network, load) |
| `pai_machine_spec.csv` | 72 KB | 1,897 | Machine specs (GPU type, CPU, memory) |

## Column Mappings

### Sensor Table (Primary)
| Raw Column | Mapped To | Description |
|------------|-----------|-------------|
| `gpu_wrk_util` | `gpu_util` | GPU utilization % |
| `cpu_usage` | `cpu_util` | CPU utilization % |
| `avg_mem` | `mem_util` | Memory utilization % |
| `avg_gpu_wrk_mem` | `gpu_mem_util` | GPU memory utilization % |

### 15 Features Generated
| # | Feature | Source | Unit |
|---|---------|--------|------|
| 0 | `facility_mw` | gpu_util → power estimation | MW |
| 1 | `power_roc` | diff(facility_mw) | MW/s |
| 2 | `power_roc2` | diff(power_roc) | MW/s² |
| 3 | `power_roll_mean` | rolling mean (window=30) | MW |
| 4 | `power_roll_std` | rolling std (window=30) | MW |
| 5 | `gpu_avg_power_norm` | gpu_util / 100 | 0-1 |
| 6 | `gpu_max_power_norm` | gpu_util / 100 | 0-1 |
| 7 | `gpu_avg_temp_norm` | estimated from util | 0-1 |
| 8 | `gpu_max_temp_norm` | estimated from util | 0-1 |
| 9 | `gpu_avg_util_norm` | gpu_util / 100 | 0-1 |
| 10 | `gpu_avg_mem_util_norm` | mem_util / 100 | 0-1 |
| 11 | `cpu_util_est_norm` | cpu_util / 100 | 0-1 |
| 12 | `hour_sin` | cyclical hour encoding | -1 to 1 |
| 13 | `hour_cos` | cyclical hour encoding | -1 to 1 |
| 14 | `is_allreduce` | heuristic (util>80 & mem<30) | 0 or 1 |

## Power Estimation Formula
```python
# GPU power model: idle 70W → peak 700W (TDP)
single_gpu_power_w = 70 + (700 - 70) * (gpu_util / 100.0)

# Facility scaling: 200,000 GPUs
facility_mw = single_gpu_power_w * 200000 / 1e6
```

## Download URLs
```
Sensor:  https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz
Metric:  https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz
Spec:    https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz
```

## Kaggle Dataset
- **URL:** https://www.kaggle.com/datasets/vedkumr/energivanu-training-data
- **Contents:** `alibaba_full.csv.gz` (413MB compressed, 50 lakh rows processed features)
- **Also:** `alibaba_300k.csv` (65MB, 3 lakh rows subset)

## GPU Types in Dataset
| GPU Type | Count | % |
|----------|-------|---|
| P100 | 798 | 42% |
| T4 | 497 | 26% |
| MISC | 280 | 15% |
| V100M32 | 135 | 7% |
| V100 | 104 | 5% |
| CPU | 83 | 4% |
