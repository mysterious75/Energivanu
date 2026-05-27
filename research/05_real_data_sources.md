# Real Data Sources for GPU Power Forecasting

> ENERGIVANU -- Research Document 05
> Goal: Replace synthetic data with real-world telemetry to eliminate overfitting
> Current bottleneck: Smooth synthetic data causes model to memorize, not generalize

---

## Table of Contents

1. [NVIDIA DCGM (Data Center GPU Manager)](#1-nvidia-dcgm)
2. [MIT Supercloud Dataset](#2-mit-supercloud-dataset)
3. [NOAA Weather Data](#3-noaa-weather-data)
4. [Grid and SCADA Data](#4-grid-and-scada-data)
5. [Other Public GPU/DC Datasets](#5-other-public-gpudc-datasets)
6. [Synthetic Data Improvement Strategies](#6-synthetic-data-improvement-strategies)
7. [Data Augmentation for Time Series](#7-data-augmentation-for-time-series)
8. [Implementation Priority Matrix](#8-implementation-priority-matrix)

---

## 1. NVIDIA DCGM

### What is DCGM

NVIDIA Data Center GPU Manager (DCGM) is a free, lightweight suite of tools for managing and monitoring NVIDIA GPUs in cluster and data center environments. It provides deep telemetry, health monitoring, diagnostics, and policy management.

- GitHub: https://github.com/NVIDIA/DCGM
- Exporter: https://github.com/NVIDIA/dcgm-exporter
- Docs: https://docs.nvidia.com/datacenter/dcgm/latest/
- Grafana Dashboard: https://grafana.com/grafana/dashboards/12239

### Metrics Available

DCGM exposes hundreds of field identifiers (DCGM_FI_*) organized by category.

#### Power Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 155 | `DCGM_FI_DEV_POWER_USAGE` | Power usage in Watts |
| 157 | `DCGM_FI_DEV_POWER_USAGE_INSTANT` | Instantaneous power in Watts |
| 156 | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | Cumulative energy in mJ since driver reload |
| 160 | `DCGM_FI_DEV_POWER_MGMT_LIMIT` | Current power cap |
| 161 | `DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN` | Minimum power limit |
| 162 | `DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX` | Maximum power limit |
| 164 | `DCGM_FI_DEV_ENFORCED_POWER_LIMIT` | Effective power limit after all limiters |
| 240 | `DCGM_FI_DEV_POWER_VIOLATION` | Power violation time in ns |

#### Temperature Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 140 | `DCGM_FI_DEV_MEMORY_TEMP` | Memory temperature in C |
| 150 | `DCGM_FI_DEV_GPU_TEMP` | GPU die temperature in C |
| 151 | `DCGM_FI_DEV_MEM_MAX_OP_TEMP` | Max memory operating temp (slowdown above) |
| 152 | `DCGM_FI_DEV_GPU_MAX_OP_TEMP` | Max GPU operating temp |
| 153 | `DCGM_FI_DEV_GPU_TEMP_LIMIT` | Thermal margin (distance to slowdown) |
| 158 | `DCGM_FI_DEV_SLOWDOWN_TEMP` | Slowdown threshold |
| 159 | `DCGM_FI_DEV_SHUTDOWN_TEMP` | Shutdown threshold |
| 241 | `DCGM_FI_DEV_THERMAL_VIOLATION` | Thermal violation time in ns |

#### Utilization Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 203 | `DCGM_FI_DEV_GPU_UTIL` | GPU utilization percentage |
| 204 | `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory copy utilization |
| 206 | `DCGM_FI_DEV_ENC_UTIL` | Encoder utilization |
| 207 | `DCGM_FI_DEV_DEC_UTIL` | Decoder utilization |
| 1001 | `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | Graphics engine active ratio |
| 1002 | `DCGM_FI_PROF_SM_ACTIVE` | SM active ratio (at least 1 warp assigned) |
| 1003 | `DCGM_FI_PROF_SM_OCCUPANCY` | Resident warps / theoretical max |
| 1004 | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor core activity ratio |
| 1005 | `DCGM_FI_PROF_DRAM_ACTIVE` | DRAM interface activity ratio |
| 1006 | `DCGM_FI_PROF_PIPE_FP64_ACTIVE` | FP64 pipe activity |
| 1007 | `DCGM_FI_PROF_PIPE_FP32_ACTIVE` | FP32 pipe activity |
| 1008 | `DCGM_FI_PROF_PIPE_FP16_ACTIVE` | FP16 pipe activity |

#### Memory Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 250 | `DCGM_FI_DEV_FB_TOTAL` | Total framebuffer in MB |
| 251 | `DCGM_FI_DEV_FB_FREE` | Free framebuffer in MB |
| 252 | `DCGM_FI_DEV_FB_USED` | Used framebuffer in MB |
| 254 | `DCGM_FI_DEV_FB_USED_PERCENT` | Framebuffer usage fraction (0.0 to 1.0) |
| 90 | `DCGM_FI_DEV_BAR1_TOTAL` | Total BAR1 memory in MB |
| 92 | `DCGM_FI_DEV_BAR1_USED` | Used BAR1 memory in MB |

#### Clock Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 100 | `DCGM_FI_DEV_SM_CLOCK` | SM clock in MHz |
| 101 | `DCGM_FI_DEV_MEM_CLOCK` | Memory clock in MHz |
| 113 | `DCGM_FI_DEV_MAX_SM_CLOCK` | Max supported SM clock |
| 114 | `DCGM_FI_DEV_MAX_MEM_CLOCK` | Max supported memory clock |

#### PCIe Metrics

| Field ID | Constant | Description |
|----------|----------|-------------|
| 235 | `DCGM_FI_DEV_PCIE_MAX_LINK_GEN` | Max PCIe link generation |
| 237 | `DCGM_FI_DEV_PCIE_LINK_GEN` | Current PCIe link generation |
| 1009 | `DCGM_FI_PROF_PCIE_TX_BYTES` | Bytes transmitted (GPU to Host) |
| 1010 | `DCGM_FI_PROF_PCIE_RX_BYTES` | Bytes received (Host to GPU) |

### Installation

#### Ubuntu/Debian

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install DCGM
sudo apt-get install -y datacenter-gpu-manager

# Start the daemon
sudo systemctl start nvidia-dcgm
sudo systemctl enable nvidia-dcgm
```

#### Docker (dcgm-exporter)

```bash
# Run dcgm-exporter as a container (simplest method)
docker run -d --gpus all --cap-add SYS_ADMIN --rm \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.5.3-4.8.2-distroless

# Verify metrics are exposed
curl -sL http://localhost:9400/metrics | head -50
```

#### Kubernetes (Helm)

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter
```

### CLI Usage (dcgmi)

```bash
# List all GPUs
dcgmi discovery -l

# Run health checks
dcgmi health -s a          # Set health checks for all GPUs
dcgmi health -r a          # Run health checks

# Run diagnostics
dcgmi diag -r 1            # Level 1 (quick, ~1 min)
dcgmi diag -r 2            # Level 2 (medium, ~5 min)
dcgmi diag -r 3            # Level 3 (extended, ~15 min)

# Create field group for power + temp monitoring
dcgmi fieldgroup -c power_temp -f 155,150,140

# Monitor with 1-second interval
dcgmi mon -c power_temp -e 155,150,140
dcgmi mon -s 1000          # 1000ms = 1 second interval
dcgmi mon -v               # View current stats

# View GPU stats
dcgmi stats -e 0           # Enable stats for GPU 0
dcgmi stats -v 0           # View stats for GPU 0
dcgmi stats -d 0           # Disable stats collection
```

### Exporting Telemetry Data

#### Method 1: Prometheus + Grafana (Recommended for Production)

```yaml
# prometheus.yml scrape config
scrape_configs:
  - job_name: 'dcgm-exporter'
    scrape_interval: 1s    # 1Hz sampling
    static_configs:
      - targets: ['dcgm-host:9400']
```

Then query from Prometheus using PromQL:
```
# Power usage over time
DCGM_FI_DEV_POWER_USAGE{gpu="0"}

# Temperature rate of change
rate(DCGM_FI_DEV_GPU_TEMP[30s])
```

Export from Prometheus to CSV via API:
```bash
curl 'http://prometheus:9090/api/v1/query_range?query=DCGM_FI_DEV_POWER_USAGE&start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z&step=1s' > power_data.json
```

#### Method 2: Custom CSV via dcgmi CLI

```bash
# Record telemetry to file
dcgmi stats -e 0
# ... run workload ...
dcgmi stats -v 0 > gpu_stats.csv
```

#### Method 3: Python API (pydcgm)

```python
import dcgm_client_api as dcgm

# Connect to DCGM host engine
dcgmHandle = dcgm.dcgmInit()
gpuIds = dcgm.dcgmGetAllSupportedDevices(dcgmHandle)

# Watch fields at specific frequency
fieldIds = [155, 150, 203]  # power, temp, util
dcgm.dcgmWatchFields(dcgmHandle, gpuIds[0], fieldIds, 100000, 5.0, 0)

# Get latest values
values = dcgm.dcgmGetLatestValues(dcgmHandle, gpuIds[0], fieldIds)
```

### Sampling Rates

| Use Case | Recommended Rate | Notes |
|----------|-----------------|-------|
| Real-time monitoring | 100ms (10Hz) | High overhead, short bursts only |
| Production telemetry | 1s (1Hz) | Good balance of detail vs overhead |
| Historical logging | 10s-60s | Low overhead, sufficient for trends |
| Power profiling | 1ms-10ms | Requires proprietary profiling mode |

The DCGM documentation recommends a minimum watch frequency of 1 second for bind/unbind events (field ID 6). Power and temperature fields support sub-second sampling but the overhead increases linearly.

### ENERGIVANU Relevance

DCGM provides the exact metrics our synthetic generator simulates:
- `DCGM_FI_DEV_POWER_USAGE` maps to `gpu_power_mw`
- `DCGM_FI_DEV_GPU_UTIL` maps to `gpu_load_pct`
- `DCGM_FI_DEV_GPU_TEMP` maps to `gpu_temp_c`
- `DCGM_FI_DEV_MEMORY_TEMP` is additional (not in synthetic)
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` captures AI workload-specific activity

To collect real data: run dcgm-exporter on a GPU node, scrape with Prometheus at 1Hz, export to CSV, and feed into `src/data/real_data.py` pipeline.

---

## 2. MIT Supercloud Dataset

### Overview

The MIT Supercloud Dataset was released by MIT Lincoln Laboratory as part of a Datacenter Challenge. It contains real GPU telemetry from MIT's TX-Green/Supercloud HPC cluster. This dataset is already partially integrated in `src/data/real_data.py`.

### S3 Bucket Access

```
s3://mit-supercloud-dataset/datacenter-challenge/202201
```

Access requires no authentication (--no-sign-request flag):

```bash
# List available files
aws s3 ls s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/ \
  --recursive --no-sign-request

# Download a specific file
aws s3 cp s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/some_file.csv \
  ./data/ --no-sign-request

# Download all GPU files
aws s3 sync s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/ \
  ./data/gpu/ --no-sign-request
```

### Data Format

Each GPU job CSV contains per-timestamp, per-GPU readings. Expected columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float/epoch | Unix timestamp or elapsed seconds |
| `gpu_index` | int | GPU device index (0, 1, 2, ...) |
| `power_draw_W` | float | Power draw in Watts |
| `utilization_gpu_pct` | float | GPU utilization (0-100) |
| `temperature_gpu` | float | GPU temperature in Celsius |
| `utilization_memory_pct` | float | Memory utilization (0-100) |
| `memory_used_MiB` | float | Memory used in MiB |
| `memory_total_MiB` | float | Total memory in MiB |

Sampling rate: approximately 100ms (10Hz), which gives high-resolution power traces.

### Existing Integration

The file `src/data/real_data.py` already implements:

1. **`list_gpu_files()`** - Lists CSV files from S3 (sorted, newest first)
2. **`download_jobs()`** - Downloads batch of GPU CSVs to local directory
3. **`load_job_csv()`** - Parses single CSV, aggregates across GPUs per timestamp
4. **`stitch_jobs()`** - Concatenates multiple job traces with idle gaps
5. **`add_features()`** - Creates rolling features (5s, 30s, 60s windows)
6. **`prepare()`** - Full pipeline: download, stitch, featurize, create X/Y/S/D arrays

The pipeline scales single-GPU power to 150K GPUs: `power_mw = power_W * 150_000 / 1e6`

### Known Issues in Current Implementation

1. Line 199 references `lb_real` but the variable is defined as `lb` on line 196 (bug)
2. The `mem_util_pct` fallback uses `gpu_util_pct` lambda when column is missing (line 77)
3. No weather or grid features are added to real data (only GPU features)
4. The stitching gap is hardcoded to 30 seconds regardless of actual time between jobs
5. No data validation or outlier filtering

### Download Strategy for Kaggle

```python
# Pre-download on Kaggle (has fast S3 connectivity)
import subprocess

bucket = "s3://mit-supercloud-dataset/datacenter-challenge/202201"
cmd = f"aws s3 sync {bucket}/gpu/ /kaggle/working/gpu_data/ --no-sign-request"
subprocess.run(cmd, shell=True)

# Then attach as Kaggle dataset for reuse across sessions
```

### Scaling Considerations

The MIT dataset captures a small cluster. To simulate 150K GPUs:
- Use statistical aggregation (Central Limit Theorem): mean scales linearly, std scales as sqrt(n)
- Apply temporal diversity: stagger job start times across simulated nodes
- Add heterogeneity: vary GPU models (A100, H100, T4) with different power profiles

---

## 3. NOAA Weather Data

### Why Weather Data Matters

GPU cooling power scales with ambient temperature. A data center in Memphis, TN (TVA territory) experiences significant seasonal and diurnal temperature variation that affects:
- HVAC power consumption (typically 30-40% of DC power)
- GPU thermal throttling thresholds
- Cooling efficiency (PUE variation)

### NOAA Climate Data Online (CDO) API

#### Getting an API Token

1. Go to: https://www.ncdc.noaa.gov/cdo-web/token
2. Register for a free token (instant delivery via email)
3. Token is passed as header: `token: YOUR_TOKEN`

#### API Base URL

```
https://www.ncdc.noaa.gov/cdo-web/api/v2/
```

#### Memphis Station IDs

| Station | ID | Location |
|---------|-----|----------|
| Memphis International Airport | `GHCND:USW00013893` | 35.04N, 89.98W |
| Memphis, TN (City) | `CITY:US470016` | FIPS code |

#### Available Datasets

| Dataset ID | Description | Resolution |
|------------|-------------|------------|
| `GHCND` | Global Historical Climatology Network - Daily | Daily |
| `GSOM` | Global Summary of the Month | Monthly |
| `GHCNDMS` | GHCN-Daily Multivariate | Daily, multivariate |
| `NORMAL_DLY` | Climate Normals | Daily normals |
| `HOURLY` (via ISD) | Integrated Surface Database | Hourly |

#### Example API Queries

```bash
# Get daily temperature for Memphis, Jan 2026
curl -H "token: YOUR_TOKEN" \
  "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:USW00013893&startdate=2026-01-01&enddate=2026-01-31&units=standard&format=json&limit=1000&datatypeid=TMAX,TMIN,TAVG"

# Find Memphis stations
curl -H "token: YOUR_TOKEN" \
  "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations?locationid=CITY:US470016&datasetid=GHCND&limit=10"

# Get available data types
curl -H "token: YOUR_TOKEN" \
  "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes?datasetid=GHCND&limit=100"
```

#### Response Format (JSON)

```json
{
  "metadata": {
    "resultset": {
      "offset": 1,
      "count": 62,
      "limit": 25
    }
  },
  "results": [
    {
      "date": "2026-01-01T00:00:00",
      "datatype": "TMAX",
      "station": "GHCND:USW00013893",
      "attributes": ",,W,2400",
      "value": 52
    }
  ]
}
```

Values are in tenths of degrees Celsius by default (divide by 10) or in standard units if `units=standard` is specified.

#### Key Data Types for ENERGIVANU

| Data Type | Description | Unit |
|-----------|-------------|------|
| `TAVG` | Average temperature | F (standard) |
| `TMAX` | Max temperature | F (standard) |
| `TMIN` | Min temperature | F (standard) |
| `PRCP` | Precipitation | inches |
| `AWND` | Average wind speed | mph |
| `WDF2` | Direction of fastest 2-min wind | degrees |
| `WSF2` | Fastest 2-min wind speed | mph |
| `SNOW` | Snowfall | inches |
| `SNWD` | Snow depth | inches |

### Bulk Download Alternative

For offline processing, download the full station file directly:

```bash
# Direct bulk download (no API key needed)
wget https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00013893.csv
```

### Hourly Data (Integrated Surface Database)

For sub-daily alignment with GPU telemetry, use the ISD:

```bash
# ISD Lite format (pre-processed hourly)
wget https://www.ncei.noaa.gov/data/global-hourly-access/subdaily/2026/USW00013893.csv

# Or use the ISD search interface
# https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
```

ISD hourly fields include: temperature, dewpoint, wind speed/direction, pressure, precipitation, visibility, cloud cover.

### Third-Party Alternatives

| Service | URL | Free Tier | Resolution |
|---------|-----|-----------|------------|
| Visual Crossing | https://www.visualcrossing.com | 1000 calls/day | Hourly |
| Open-Meteo | https://open-meteo.com | Unlimited | Hourly |
| WeatherAPI | https://weatherapi.com | 1M calls/month | 15-min |

#### Open-Meteo (No API Key Required)

```bash
# Historical weather for Memphis (35.15N, 90.05W)
curl "https://archive-api.open-meteo.com/v1/archive?latitude=35.15&longitude=-90.05&start_date=2026-01-01&end_date=2026-01-31&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover&timezone=America/Chicago"
```

Response is JSON with hourly arrays -- easy to convert to DataFrame.

### Timestamp Alignment

To align weather data with GPU telemetry:

```python
import pandas as pd

# GPU data (5-second intervals from synthetic or 100ms from DCGM)
gpu_df = pd.read_csv("gpu_telemetry.csv", parse_dates=["timestamp"])

# Weather data (hourly from NOAA or Open-Meteo)
weather_df = pd.read_csv("weather.csv", parse_dates=["timestamp"])

# Forward-fill weather to match GPU timestamps
weather_df = weather_df.set_index("timestamp").resample("5s").ffill()

# Merge
merged = gpu_df.merge(weather_df, on="timestamp", how="left")
```

For daily NOAA data, use interpolation instead of forward-fill to avoid step-function artifacts.

---

## 4. Grid and SCADA Data

### TVA (Tennessee Valley Authority)

TVA is the largest public power provider in the US, serving ~10 million people across 7 southeastern states. A data center in Memphis would be served by TVA.

#### What is Publicly Available

Real-time SCADA data (voltage, frequency, breaker status) is classified as BES Cyber System Information under NERC CIP and is restricted for national security. However, aggregated operational data is available through:

#### EIA (Energy Information Administration)

- **API**: https://www.eia.gov/opendata/
- **Grid Monitor**: https://www.eia.gov/electricity/gridmonitor/
- **API Key**: Register at https://www.eia.gov/opendata/ (free)

```bash
# Get TVA electricity data
curl "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key=YOUR_KEY&frequency=hourly&data[0]=value&facets[respondent][]=TVA&start=2026-01-01&end=2026-01-31"
```

EIA provides hourly generation by fuel type for TVA including:
- Coal, Natural Gas, Nuclear, Hydro, Solar, Wind, Other
- Total demand and generation
- Interchange with neighboring balancing authorities

#### TVA Direct Resources

| Resource | URL | Data Available |
|----------|-----|----------------|
| TVA Power System | https://www.tva.com/energy/our-power-system | Generation mix, capacity |
| TVA IRP | https://www.tva.com/energy/integrated-resource-plan | Load projections |
| TVA Open Data (limited) | Search "TVA data" on data.gov | Historical generation |

#### EIA Grid Monitor Data

The EIA Grid Monitor provides near real-time data by balancing authority:

```
https://www.eia.gov/electricity/gridmonitor/
```

Available fields (hourly):
- Demand (MW)
- Generation (MW)
- Net interchange (MW)
- Generation by fuel type
- CO2 emissions estimates

### Grid Frequency Data

Real-time grid frequency (60 Hz nominal) is useful as a stress indicator.

| Source | URL | Resolution | Cost |
|--------|-----|------------|------|
| GridStatus.io | https://gridstatus.io | 5-min to 1-sec | Free tier available |
| FNET/GridEye | https://fnet.utk.edu | 10-sec | Free (academic) |
| ERCOT | https://ercot.com | 5-min | Free |
| PJM | https://pjm.com | 5-min | Free (registration) |
| NERC | https://nerc.com | Varies | Academic access |

#### GridStatus.io API

```python
# Python client
pip install gridstatus

import gridstatus
iso = gridstatus.TVA()  # If available, otherwise use MISO or SPP
data = iso.get_fuel_mix(date="2026-01-01")
```

Note: TVA is not an ISO/RTO, so direct API access is limited. The closest ISOs are MISO (Midcontinent) and SPP (Southwest Power Pool), which may cover parts of TVA's interconnection.

### Making Grid Data Realistic Without Real-Time Access

If real grid data is unavailable, improve the synthetic grid model:

```python
# Realistic grid frequency model
# Based on: frequency deviation = (generation - demand) / (2 * H * f0)
# where H = inertia constant (~5 seconds), f0 = 60 Hz

def realistic_frequency(power_demand_mw, nominal_mw=1000, inertia_h=5.0):
    """Model frequency deviation from power imbalance."""
    imbalance = power_demand_mw - nominal_mw
    deviation = imbalance / (2 * inertia_h * 60)
    # Add measurement noise + AGC oscillation
    noise = np.random.normal(0, 0.008)  # ~8 mHz noise
    agc = 0.01 * np.sin(2 * np.pi * np.arange(len(power_demand_mw)) / 300)
    return 60.0 + deviation + noise + agc
```

### EIA Historical Data Download

For training data, bulk download historical TVA data:

```python
import requests

API_KEY = "your_eia_api_key"
url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
params = {
    "api_key": API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "TVA",
    "start": "2020-01-01",
    "end": "2026-01-01",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
}
resp = requests.get(url, params=params)
data = resp.json()
```

---

## 5. Other Public GPU/DC Datasets

### Google Cluster Data

Repository: https://github.com/google/cluster-data

#### Available Traces

| Dataset | Scope | Year | Format |
|---------|-------|------|--------|
| ClusterData2011 | 12,500 machines, 29 days | 2011 | Protobuf v2 |
| ClusterData2019 | 8 Borg cells, 1 month (May) | 2019 | Protobuf v3 |
| PowerData2019 | 57 power domains, 1 month (May) | 2019 | CSV/Protobuf |
| ETA Traces | Distributed system execution traces | Various | Custom |

#### PowerData2019 (Highly Relevant)

This is the most relevant dataset for ENERGIVANU:
- **57 power domains** measured over May 2019
- Synergistic with ClusterData2019 workload data
- Documentation: `power_trace_documentation.pdf` in repo
- Analysis notebook: `power_trace_analysis_colab.ipynb`
- License: CC-BY

Download: Data files hosted on Google Cloud Storage (GCS). Access instructions in the GitHub repo.

#### How to Use PowerData2019

```python
# After downloading from GCS
import pandas as pd

# Power traces are typically in format:
# timestamp, domain_id, power_watts
power_df = pd.read_csv("power_data.csv")

# Correlate with workload traces
# ClusterData2019 has: job_id, task_id, timestamp, cpu_usage, memory_usage, etc.
```

### Microsoft Azure Public Datasets

Repository: https://github.com/Azure/AzurePublicDataset

#### Available Datasets

| Dataset | Year | Description | Size |
|---------|------|-------------|------|
| AzurePublicDatasetV1 | 2017 | ~2M VMs, 1.2B utilization readings | Large |
| AzurePublicDatasetV2 | 2019 | ~2.6M VMs, 1.9B utilization readings | Large |
| AzureTracesForPacking | 2020 | VM requests with priorities, lifetimes | Medium |
| AzureVMNoiseDataset | 2024 | 483 days of benchmark noise data | Medium |
| AzureFunctionsDataset | 2019 | Serverless invocation traces | Medium |
| AzureFunctionsInvocationTrace | 2021 | 2-week function trace | Small |
| AzureLLMInferenceDataset2023 | 2023 | LLM inference input/output tokens | Small |
| AzureLLMInferenceDataset2024 | 2024 | 1-week LLM inference sample | Small |
| AzureLMMInferenceDataset2025 | 2025 | Multimodal inference trace | Small |
| AzureGreenSKUFramework2023 | 2023 | Carbon-aware server design data | Medium |

#### Most Relevant for ENERGIVANU

1. **AzurePublicDatasetV2** - VM utilization data can approximate GPU utilization patterns
2. **AzureLLMInferenceDataset2024** - LLM inference workloads directly use GPUs
3. **AzureGreenSKUFramework2023** - Power and carbon data for server design
4. **AzureVMNoiseDataset2024** - Performance variability data (useful for noise modeling)

Download links are in `.txt` files in the repo (Azure Blob Storage URLs).

### Alibaba Cluster Traces

Repository: https://github.com/alibaba/clusterdata

#### GPU-Specific Datasets

| Dataset | Scope | GPUs | Duration | Key Feature |
|---------|-------|------|----------|-------------|
| cluster-trace-gpu-v2020 | 6,500+ GPUs, 1,800 machines | A100/V100 | 2 months | MLaaS training workloads |
| cluster-trace-gpu-v2023 | 6,200+ GPUs, 1,200 machines | Mixed | N/A | Heterogeneous GPU sharing |
| cluster-trace-gpu-v2025 | 20K+ inference instances | Mixed | N/A | DLRM inference disaggregation |
| cluster-trace-v2026-GenAI | Stable Diffusion serving | GPU | N/A | Full stack: app/middleware/infra |
| cluster-trace-v2026-spot-gpu | Spot GPU instances | GPU | N/A | Spot/preemptible workloads |

#### cluster-trace-gpu-v2020 (Best for Training Data)

- 6,500+ GPUs across ~1,800 machines
- 2 months of production MLaaS workloads
- Published at USENIX NSDI '22: "MLaaS in the Wild"
- Includes: job IDs, task IDs, GPU utilization, memory usage, timestamps
- Data, schema, and processing scripts provided in Jupyter notebooks

#### cluster-trace-v2026-GenAI (Most Recent)

- Comprehensive trace of Stable Diffusion model serving
- Three layers captured:
  - Application layer: user requests, end-to-end latency
  - Middleware layer: gateway queues, schedulers, pipeline management
  - Infrastructure layer: container resources, GPU utilization, memory usage

#### Access

Most datasets are available directly in the repository subdirectories. Older datasets (v2017, v2018) require completing a short survey.

```bash
# Clone the repository
git clone https://github.com/alibaba/clusterdata.git

# GPU v2020 data
ls clusterdata/cluster-trace-gpu-v2020/

# GPU v2023 data
ls clusterdata/cluster-trace-gpu-v2023/
```

### NVIDIA Published Data

NVIDIA does not publish raw GPU telemetry datasets publicly. However:

1. **MLPerf Results**: https://mlcommons.org/en/ -- benchmark results including power
2. **DCGM Sample Data**: The dcgm-exporter Grafana dashboard (ID 12239) comes with sample data
3. **NVIDIA Research Papers**: Some include supplementary data (check https://research.nvidia.com)

### Bitbrains Data Center Traces

The Bitbrains dataset is a well-known data center trace:
- Source: Bitbrains (Dutch hosting provider)
- Contains: VM CPU, memory, disk I/O, network metrics
- Resolution: 5-minute intervals
- Duration: Multiple months
- Access: Was available via the DASC (Datacenter Analytics and Simulation Challenge)

### SPEC Power Benchmarks

Not time-series data, but useful for calibrating power models:
- https://www.spec.org/power/
- Server power consumption at various load levels (10%, 50%, 100%)
- Can be used to validate synthetic power curves

---

## 6. Synthetic Data Improvement Strategies

### Problem Statement

Our current synthetic data (in `src/data/generator.py`) has these issues:
1. Power curves are too smooth (simple sine waves + sparse spikes)
2. No realistic noise patterns (just Gaussian noise)
3. Features are independently generated (no realistic correlations)
4. Spikes are uniform random, not physics-based
5. No temporal autocorrelation beyond rolling means

### Strategy 1: Physics-Based Power Model

Replace the simple `power = idle + dynamic * util^1.3` with a thermal RC network model.

```python
class PhysicsBasedGPUPower:
    """GPU power model with thermal feedback and DVFS."""

    def __init__(self, gpu_tdp=700, gpu_idle=70, thermal_mass=500,
                 thermal_resistance=0.1, ambient_temp=25):
        self.tdp = gpu_tdp        # Watts
        self.idle = gpu_idle      # Watts
        self.C = thermal_mass     # J/K (thermal capacitance)
        self.R = thermal_resistance  # K/W (thermal resistance)
        self.T_amb = ambient_temp
        self.T_gpu = ambient_temp
        self.dt = 5.0             # 5-second timestep

    def step(self, utilization):
        """Single timestep: compute power and temperature."""
        # Dynamic power with voltage-frequency scaling
        # P = C * V^2 * f, where V scales with f
        # Approximate: P_dynamic = P_max * util^alpha * (1 - thermal_throttle)
        alpha = 1.35  # Empirical exponent for GPU workloads

        # Thermal throttling: reduce clock when near TDP temp
        T_limit = 83.0  # Typical GPU throttle temp
        throttle = max(0, (self.T_gpu - T_limit + 5) / 5)
        throttle = min(1, throttle)  # 0 = no throttle, 1 = full throttle

        dynamic = (self.tdp - self.idle) * (utilization ** alpha) * (1 - 0.3 * throttle)
        power = self.idle + dynamic

        # Thermal dynamics: dT/dt = (P - (T - T_amb)/R) / C
        dTdt = (power - (self.T_gpu - self.T_amb) / self.R) / self.C
        self.T_gpu += dTdt * self.dt

        # Add measurement noise (ADC quantization, sensor noise)
        power_noisy = power + np.random.normal(0, power * 0.005)  # 0.5% noise
        temp_noisy = self.T_gpu + np.random.normal(0, 0.3)  # 0.3C noise

        return power_noisy, temp_noisy, throttle
```

### Strategy 2: Realistic Workload Patterns

Replace random spikes with patterns derived from real HPC workload analysis.

```python
class RealisticWorkload:
    """Generate workload patterns based on real data center behavior."""

    def generate_job_arrival(self, n_steps, interval_sec=5):
        """Model job arrivals as a non-homogeneous Poisson process."""
        t = np.arange(n_steps) * interval_sec / 3600  # hours

        # Base rate: higher during business hours
        base_rate = 0.3 + 0.4 * np.exp(-((t % 24 - 14) / 4) ** 2)

        # Batch job scheduling (every 4 hours, as in SLURM clusters)
        batch_rate = 0.2 * (np.abs(np.sin(np.pi * t / 2)) < 0.1).astype(float)

        # Weekend reduction
        dow = (t / 24).astype(int) % 7
        weekend_factor = np.where(dow >= 5, 0.6, 1.0)

        total_rate = (base_rate + batch_rate) * weekend_factor

        # Generate arrivals
        arrivals = np.random.poisson(total_rate)
        return arrivals

    def generate_utilization_profile(self, arrivals, n_steps):
        """Convert arrivals to GPU utilization with realistic shapes."""
        util = np.ones(n_steps) * 0.15  # Idle baseline

        for i in range(n_steps):
            if arrivals[i] > 0:
                # Job startup ramp (30-120 seconds to reach full utilization)
                ramp_time = np.random.randint(6, 24)  # in 5-sec steps
                # Job duration (heavy-tailed: most short, some very long)
                duration = int(np.random.lognormal(4, 1.5))  # ~55 steps median
                duration = min(duration, n_steps - i)

                # Utilization shape: ramp up, sustained, ramp down
                for j in range(duration):
                    if i + j >= n_steps:
                        break
                    if j < ramp_time:
                        phase_util = 0.7 + 0.3 * (j / ramp_time)
                    elif j > duration - ramp_time:
                        phase_util = 0.7 + 0.3 * ((duration - j) / ramp_time)
                    else:
                        phase_util = 1.0

                    # Add per-job noise (memory access patterns, etc.)
                    job_noise = np.random.normal(0, 0.05)
                    util[i + j] = max(util[i + j], min(1.0, phase_util + job_noise))

        return util
```

### Strategy 3: Feature Correlations

Model realistic correlations between features rather than generating them independently.

```python
def correlated_features(util, temp_ambient, wind_speed, cloud_cover):
    """Generate correlated features from base utilization."""
    n = len(util)

    # GPU temperature follows utilization with thermal lag
    gpu_temp = thermal_model(util, temp_ambient)  # Uses RC network above

    # Power correlates with temperature (leakage current increases with temp)
    leakage_factor = 1 + 0.005 * (gpu_temp - 25)  # ~0.5% per degree
    power = base_power(util) * leakage_factor

    # Memory utilization correlates with compute but has different pattern
    mem_util = util * 0.7 + 0.3 * np.random.beta(2, 5, n)  # skewed

    # PCIe bandwidth correlates with memory util
    pcie_bw = mem_util * 0.8 + np.random.normal(0, 0.05, n)

    # Clock frequency: DVFS responds to utilization
    # Higher util -> higher clock, but throttles at high temp
    clock = 1000 + 1500 * util - 500 * np.clip((gpu_temp - 80) / 10, 0, 1)

    return {
        "power": power,
        "gpu_temp": gpu_temp,
        "mem_util": mem_util,
        "pcie_bw": pcie_bw,
        "clock_mhz": clock,
    }
```

### Strategy 4: Realistic Noise Patterns

Replace Gaussian noise with structured noise found in real measurements.

```python
class RealisticNoise:
    """Noise models based on real sensor characteristics."""

    @staticmethod
    def power_noise(n, base_power):
        """Power measurement noise: ADC quantization + 1/f noise."""
        # ADC quantization (12-bit ADC, 0-1000W range)
        quant_step = 1000 / 4096
        quantized = np.round(base_power / quant_step) * quant_step

        # 1/f (pink) noise from power supply ripple
        pink = generate_pink_noise(n, amplitude=base_power * 0.002)

        # Occasional measurement glitches (0.1% of samples)
        glitches = np.zeros(n)
        glitch_idx = np.random.choice(n, size=n // 1000, replace=False)
        glitches[glitch_idx] = np.random.normal(0, base_power * 0.05)

        return quantized + pink + glitches

    @staticmethod
    def temperature_noise(n):
        """Temperature sensor noise: slow drift + quantization."""
        # Sensor drift (random walk)
        drift = np.cumsum(np.random.normal(0, 0.01, n))

        # Quantization (0.5C steps for many sensors)
        # Applied externally

        # Brownian component (thermal mass filtering)
        brownian = np.cumsum(np.random.normal(0, 0.005, n))

        return drift + brownian

def generate_pink_noise(n, amplitude=1.0):
    """Generate 1/f noise using Voss-McCartney algorithm."""
    # Simplified: use filtered white noise
    white = np.random.randn(n)
    # Apply 1/f filter via FFT
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # avoid division by zero
    fft = np.fft.rfft(white)
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n)
    return pink * amplitude / np.std(pink)
```

### Strategy 5: Spike Simulation with Proper Physics

```python
class PowerSpike:
    """Simulate realistic power spikes based on physical causes."""

    @staticmethod
    def job_start(n_gpus=8, max_power_per_gpu=700):
        """Job initialization: memory allocation, kernel compilation."""
        # Phase 1: Memory allocation (100-500ms) - moderate power
        # Phase 2: Kernel compilation (1-5s) - high CPU, moderate GPU
        # Phase 3: Data loading (5-30s) - high PCIe, moderate GPU
        # Phase 4: First kernel launch - full GPU power
        phases = [
            {"duration": (20, 100), "util": (0.3, 0.5)},   # memory alloc
            {"duration": (200, 1000), "util": (0.5, 0.8)},  # compilation
            {"duration": (1000, 6000), "util": (0.6, 0.9)}, # data load
        ]
        return phases

    @staticmethod
    def thermal_throttle(gpu_temp, throttle_temp=83, recovery_temp=75):
        """Model thermal throttling behavior."""
        if gpu_temp >= throttle_temp:
            # Throttle: reduce clock -> reduce power
            throttle_pct = min(1.0, (gpu_temp - throttle_temp) / 5)
            return throttle_pct
        elif gpu_temp > recovery_temp:
            # Hysteresis: don't fully recover until well below threshold
            return max(0, (gpu_temp - recovery_temp) / (throttle_temp - recovery_temp))
        return 0.0

    @staticmethod
    def cascade_failure(n_gpus, trigger_gpu=0):
        """Model cascading thermal events across GPUs in a node."""
        # When one GPU overheats, adjacent GPUs get less airflow
        # This can trigger a cascade
        cascade = np.zeros(n_gpus)
        cascade[trigger_gpu] = 1.0

        for t in range(1, 60):  # 60 timesteps
            for g in range(n_gpus):
                if cascade[g] > 0:
                    # Heat spreads to neighbors
                    for neighbor in [g-1, g+1]:
                        if 0 <= neighbor < n_gpus:
                            cascade[neighbor] += 0.05 * cascade[g]
            cascade = np.clip(cascade, 0, 1)

        return cascade
```

### Validation Strategy

To validate synthetic data quality, compare statistical properties against real data:

```python
def validate_synthetic(real_df, synth_df, features):
    """Compare statistical properties of real vs synthetic data."""
    report = {}
    for feat in features:
        r = real_df[feat].dropna()
        s = synth_df[feat].dropna()

        report[feat] = {
            # Distribution similarity
            "ks_stat": ks_2samp(r, s).statistic,
            "mean_diff_pct": abs(r.mean() - s.mean()) / r.mean() * 100,
            "std_diff_pct": abs(r.std() - s.std()) / r.std() * 100,

            # Temporal properties
            "autocorr_real": r.autocorr(lag=1),
            "autocorr_synth": s.autocorr(lag=1),

            # Extreme values
            "p99_real": r.quantile(0.99),
            "p99_synth": s.quantile(0.99),
            "max_ratio": s.max() / r.max(),
        }

    # Cross-correlation between features
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            real_corr = real_df[f1].corr(real_df[f2])
            synth_corr = synth_df[f1].corr(synth_df[f2])
            report[f"corr_{f1}_{f2}"] = {
                "real": real_corr,
                "synthetic": synth_corr,
                "diff": abs(real_corr - synth_corr),
            }

    return report
```

---

## 7. Data Augmentation for Time Series

### Overview

Data augmentation increases effective training set size without collecting new data. For time series forecasting, augmentation must preserve temporal structure.

### Technique 1: Jittering (Noise Injection)

Add small random noise to each sample independently.

```python
def jittering(X, sigma=0.03):
    """Add Gaussian noise to time series.
    Args:
        X: shape (batch, seq_len, features)
        sigma: noise standard deviation as fraction of signal std
    Returns:
        Augmented X with same shape
    """
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise * X.std(axis=1, keepdims=True)
```

**Why it helps**: Makes model robust to sensor measurement noise. Especially important for power measurements which have ADC quantization noise.

### Technique 2: Scaling (Magnitude Scaling)

Multiply entire time series by a random scalar.

```python
def scaling(X, sigma=0.1):
    """Scale entire series by random factor.
    Args:
        X: shape (batch, seq_len, features)
        sigma: std of scaling factor (centered on 1.0)
    Returns:
        Scaled X
    """
    factors = np.random.normal(1.0, sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * factors
```

**Why it helps**: Simulates variations in cluster size, GPU count, or workload intensity.

### Technique 3: Window Slicing

Extract random sub-sequences from the input window.

```python
def window_slicing(X, Y, slice_ratio=0.9):
    """Extract random sub-window from time series.
    Args:
        X: shape (batch, seq_len, features)
        Y: shape (batch, horizon)
        slice_ratio: fraction of original length to keep
    Returns:
        Sliced X (shorter), Y unchanged
    """
    batch, seq_len, feat = X.shape
    slice_len = int(seq_len * slice_ratio)
    start = np.random.randint(0, seq_len - slice_len + 1, size=batch)

    X_sliced = np.array([X[i, start[i]:start[i]+slice_len] for i in range(batch)])
    return X_sliced, Y
```

**Why it helps**: Forces model to make predictions from different temporal contexts, improving robustness to missing data.

### Technique 4: Magnitude Warping

Smoothly warp the magnitude of the time series using a cubic spline.

```python
def magnitude_warping(X, sigma=0.2, knot=4):
    """Warp magnitude using smooth spline curve.
    Args:
        X: shape (batch, seq_len, features)
        sigma: warping intensity
        knot: number of control points for spline
    Returns:
        Warped X
    """
    from scipy.interpolate import CubicSpline

    batch, seq_len, feat = X.shape
    X_warped = X.copy()

    for i in range(batch):
        for f in range(feat):
            # Random control points
            knot_points = np.linspace(0, seq_len - 1, knot + 2)
            knot_values = np.random.normal(1.0, sigma, knot + 2)

            # Smooth warp curve
            cs = CubicSpline(knot_points, knot_values)
            warp_curve = cs(np.arange(seq_len))

            X_warped[i, :, f] *= warp_curve

    return X_warped
```

**Why it helps**: Creates realistic variations where different parts of the time series are amplified differently, simulating workload phase transitions.

### Technique 5: Time Warping

Distort the temporal axis to simulate speed variations.

```python
def time_warping(X, sigma=0.2, knot=4):
    """Warp time axis using smooth spline.
    Args:
        X: shape (batch, seq_len, features)
        sigma: warping intensity
        knot: number of control points
    Returns:
        Time-warped X (same shape, interpolated)
    """
    from scipy.interpolate import CubicSpline

    batch, seq_len, feat = X.shape
    X_warped = X.copy()

    for i in range(batch):
        # Random time distortion
        knot_points = np.linspace(0, seq_len - 1, knot + 2)
        knot_offsets = np.random.normal(0, sigma * seq_len / knot, knot + 2)
        knot_offsets[0] = 0
        knot_offsets[-1] = 0

        # Cumulative sum to ensure monotonicity
        warped_knots = knot_points + np.cumsum(knot_offsets)
        warped_knots = np.clip(warped_knots, 0, seq_len - 1)

        cs = CubicSpline(warped_knots, knot_points)
        new_indices = cs(np.arange(seq_len))
        new_indices = np.clip(new_indices, 0, seq_len - 1)

        for f in range(feat):
            X_warped[i, :, f] = np.interp(
                np.arange(seq_len), new_indices, X[i, :, f]
            )

    return X_warped
```

**Why it helps**: Simulates variations in event timing (e.g., job start delays, thermal lag variations).

### Technique 6: Window Warping

Combine window slicing with time warping for more aggressive augmentation.

```python
def window_warping(X, window_ratio=0.5, scales=[0.5, 2.0]):
    """Warp a random window by a random time scale.
    Args:
        X: shape (batch, seq_len, features)
        window_ratio: fraction of sequence to warp
        scales: possible time scale factors
    """
    batch, seq_len, feat = X.shape
    X_warped = X.copy()

    for i in range(batch):
        window_len = int(seq_len * window_ratio)
        start = np.random.randint(0, seq_len - window_len)
        scale = np.random.choice(scales)

        # Resample the window
        original_indices = np.arange(window_len)
        new_len = int(window_len * scale)
        new_indices = np.linspace(0, window_len - 1, new_len)
        new_indices = np.clip(new_indices, 0, window_len - 1)

        for f in range(feat):
            warped_window = np.interp(
                new_indices, original_indices,
                X[i, start:start+window_len, f]
            )

            # Replace in original sequence (trim or pad as needed)
            if scale < 1:
                # Shorter: pad with last value
                X_warped[i, start:start+new_len, f] = warped_window
                X_warped[i, start+new_len:start+window_len, f] = warped_window[-1]
            else:
                # Longer: truncate
                X_warped[i, start:start+window_len, f] = warped_window[:window_len]

    return X_warped
```

### Technique 7: DBA (DTW Barycentric Averaging)

Generate synthetic series by averaging time-warped versions of existing series.

```python
def dba_augmentation(X, n_augmented=3, n_iterations=10):
    """Generate augmented series via DTW Barycentric Averaging.
    Requires: dtw-python or tslearn
    """
    from tslearn.barycenters import dtw_barycenter_averaging

    batch, seq_len, feat = X.shape
    augmented = []

    for i in range(min(n_augmented, batch)):
        # Select random subset to average
        indices = np.random.choice(batch, size=min(5, batch), replace=False)
        subset = X[indices]  # (5, seq_len, feat)

        # DBA per feature
        for f in range(feat):
            series_list = [subset[j, :, f] for j in range(len(indices))]
            barycenter = dtw_barycenter_averaging(
                series_list, max_iter=n_iterations
            )
            augmented.append(barycenter)

    return np.array(augmented).reshape(-1, seq_len, feat)
```

### Combined Augmentation Pipeline

```python
class AugmentationPipeline:
    """Apply multiple augmentations with configurable probabilities."""

    def __init__(self):
        self.augmentations = [
            (jittering, 0.8, {"sigma": 0.03}),
            (scaling, 0.5, {"sigma": 0.1}),
            (magnitude_warping, 0.3, {"sigma": 0.2, "knot": 4}),
            (time_warping, 0.3, {"sigma": 0.2, "knot": 4}),
        ]

    def __call__(self, X, Y):
        """Apply augmentations to a batch."""
        X_aug = X.copy()
        Y_aug = Y.copy()

        for aug_fn, prob, kwargs in self.augmentations:
            if np.random.random() < prob:
                X_aug = aug_fn(X_aug, **kwargs)

        return X_aug, Y_aug
```

### Augmentation Guidelines for ENERGIVANU

| Technique | Priority | Risk | When to Use |
|-----------|----------|------|-------------|
| Jittering | High | Low | Always -- simulates sensor noise |
| Scaling | High | Low | Always -- simulates cluster size variation |
| Window Slicing | Medium | Medium | When overfitting on specific patterns |
| Magnitude Warping | Medium | Medium | When model fails on phase transitions |
| Time Warping | Low | High | Only if temporal alignment is not critical |
| DBA | Low | High | Only with sufficient real data as seed |

**Caution**: Augmentation on synthetic data compounds the synthetic-ness. It is better to augment real data than to augment synthetic data.

---

## 8. Implementation Priority Matrix

### Priority 1: Quick Wins (Do First)

| Action | Effort | Impact | Files to Change |
|--------|--------|--------|-----------------|
| Fix `real_data.py` bug (line 199: `lb_real` -> `lb`) | 5 min | High | `src/data/real_data.py` |
| Add jittering + scaling augmentation | 1 hour | Medium | New: `src/data/augment.py` |
| Download Open-Meteo weather for Memphis | 30 min | Medium | New: `data/weather/` |
| Download EIA TVA hourly data | 30 min | Medium | New: `data/grid/` |

### Priority 2: Real Data Integration (1-2 Days)

| Action | Effort | Impact | Files to Change |
|--------|--------|--------|-----------------|
| Download Alibaba GPU-v2020 trace | 2 hours | High | New: `data/alibaba_gpu/` |
| Process Alibaba trace into X/Y format | 4 hours | High | `src/data/real_data.py` |
| Add weather features to real data pipeline | 2 hours | Medium | `src/data/real_data.py` |
| Add grid features to real data pipeline | 2 hours | Medium | `src/data/real_data.py` |

### Priority 3: Synthetic Data Improvement (2-3 Days)

| Action | Effort | Impact | Files to Change |
|--------|--------|--------|-----------------|
| Implement physics-based GPU power model | 4 hours | High | `src/data/generator.py` |
| Add realistic workload patterns | 3 hours | High | `src/data/generator.py` |
| Add feature correlations | 2 hours | Medium | `src/data/generator.py` |
| Implement realistic noise models | 2 hours | Medium | `src/data/generator.py` |
| Add validation against real data | 3 hours | Medium | New: `src/data/validate.py` |

### Priority 4: DCGM Collection (Requires GPU Hardware)

| Action | Effort | Impact | Files to Change |
|--------|--------|--------|-----------------|
| Set up dcgm-exporter on GPU node | 1 hour | High | Deployment config |
| Configure Prometheus scraping at 1Hz | 1 hour | High | `prometheus.yml` |
| Run workload and collect 24h of data | 24 hours | High | Data collection |
| Export and process DCGM data | 2 hours | High | `src/data/real_data.py` |

### Priority 5: Additional Augmentation (As Needed)

| Action | Effort | Impact | Files to Change |
|--------|--------|--------|-----------------|
| Implement all 7 augmentation techniques | 4 hours | Medium | `src/data/augment.py` |
| Add augmentation to training loop | 1 hour | Medium | Training code |
| Tune augmentation parameters | 2 hours | Low | Config |

---

## Appendix A: Quick Reference URLs

| Resource | URL |
|----------|-----|
| NVIDIA DCGM GitHub | https://github.com/NVIDIA/DCGM |
| DCGM Exporter GitHub | https://github.com/NVIDIA/dcgm-exporter |
| DCGM Field IDs | https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html |
| DCGM Grafana Dashboard | https://grafana.com/grafana/dashboards/12239 |
| MIT Supercloud S3 | s3://mit-supercloud-dataset/datacenter-challenge/202201 |
| NOAA CDO API | https://www.ncdc.noaa.gov/cdo-web/api/v2/ |
| NOAA API Token | https://www.ncdc.noaa.gov/cdo-web/token |
| Open-Meteo Historical | https://archive-api.open-meteo.com/v1/archive |
| EIA Open Data API | https://www.eia.gov/opendata/ |
| EIA Grid Monitor | https://www.eia.gov/electricity/gridmonitor/ |
| GridStatus.io | https://gridstatus.io |
| Google Cluster Data | https://github.com/google/cluster-data |
| Google PowerData2019 | In repo: google/cluster-data under PowerData2019 |
| Azure Public Datasets | https://github.com/Azure/AzurePublicDataset |
| Alibaba Cluster Data | https://github.com/alibaba/clusterdata |
| Alibaba GPU Trace v2020 | In repo: alibaba/clusterdata/cluster-trace-gpu-v2020 |
| Alibaba GPU Trace v2023 | In repo: alibaba/clusterdata/cluster-trace-gpu-v2023 |
| Alibaba GenAI Trace v2026 | In repo: alibaba/clusterdata/cluster-trace-v2026-GenAI |

## Appendix B: Data Format Conversion Template

```python
"""
Template: Convert any real data source to ENERGIVANU format.
The target format matches src/data/generator.py output.
"""

import pandas as pd
import numpy as np

def convert_to_energivanu_format(
    timestamps: np.ndarray,      # Unix timestamps or datetime
    gpu_power_w: np.ndarray,     # Per-GPU power in Watts
    gpu_util_pct: np.ndarray,    # GPU utilization 0-100
    gpu_temp_c: np.ndarray,      # GPU temperature Celsius
    ambient_temp_c: np.ndarray,  # Ambient temperature
    humidity_pct: np.ndarray,    # Humidity 0-100
    wind_ms: np.ndarray,         # Wind speed m/s
    n_gpus: int = 150_000,       # Scale factor
    target_interval: int = 5,    # Target interval in seconds
) -> pd.DataFrame:
    """Convert real data to ENERGIVANU DataFrame format."""

    # Scale power to cluster level
    gpu_power_mw = gpu_power_w * n_gpus / 1e6

    # Build DataFrame matching generator.py output
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s"),
        "gpu_power_mw": gpu_power_mw,
        "gpu_load_pct": gpu_util_pct,
        "gpu_temp_c": gpu_temp_c,
        "temp_c": ambient_temp_c,
        "humid_pct": humidity_pct,
        "cloud_pct": np.zeros(len(timestamps)),  # Fill from weather data
        "solar_wm2": np.zeros(len(timestamps)),
        "solar_mw": np.zeros(len(timestamps)),
        "wind_ms": wind_ms,
        "freq_hz": np.full(len(timestamps), 60.0),
        "volt_pu": np.ones(len(timestamps)),
        "soc_pct": np.full(len(timestamps), 75.0),
        "batt_mw": np.zeros(len(timestamps)),
        "grid_mw": gpu_power_mw,
    })

    # Add derived features
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["dow"] = df["timestamp"].dt.dayofweek
    df["pwr_rate"] = df["gpu_power_mw"].diff().fillna(0) / target_interval
    df["net_load"] = df["gpu_power_mw"] - df["solar_mw"]

    return df
```
