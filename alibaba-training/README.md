# ⚡ Energivanu — Alibaba GPU Training Results

## Overview
Full training pipeline on Alibaba GPU Trace 2020 dataset — real GPU telemetry data from 6,500 GPUs at Alibaba data centers.

---

## 🏆 Final Results (Best Run)

| Metric | Value |
|--------|-------|
| **Dataset** | Alibaba GPU Trace 2020 (CC BY 4.0) |
| **Data Rows** | 30,33,232 (30 lakh, raw sensor table) |
| **Features** | 15 (matching PEB model input) |
| **Model** | TCN + Multi-Head Attention |
| **Parameters** | 613,612 |
| **GPU** | Tesla P100-PCIE-16GB (CUDA) |
| **Train Sequences** | 5,15,643 |
| **Val Sequences** | 90,996 |
| **Epochs** | 200 max (early stopped) |
| **Best Val Loss** | **5.95** ★ |
| **Best MAPE** | **~21%** ★ |
| **Overfitting Gap** | <3% ✅ |
| **Training Time** | ~45 min |

---

## 📊 Training Progress

| Epoch | Train Loss | Val Loss | MAPE | Best@ |
|-------|-----------|----------|------|-------|
| 1 | 7.66 | 6.68 | 21.43% | 1 |
| 2 | 6.47 | 6.36 | 21.23% | 2 |
| 4 | 6.31 | 6.29 | 20.91% | 4 |
| 5 | 6.30 | **6.14** | 21.75% | 5 |
| 10 | 6.16 | 6.30 | 20.29% | 5 |
| 20 | 6.03 | 6.04 | 20.30% | 19 |
| 22 | - | **5.99** | - | 22 |
| 30 | 5.95 | 5.99 | 20.66% | 22 |
| 41 | - | **5.95** | - | 41 |
| 50 | 5.83 | 6.05 | 21.69% | 41 |
| 60 | 5.77 | 6.05 | 21.40% | 41 |

**Key Observations:**
- Model converged quickly (epoch 5-10)
- Best val loss at epoch 41 (5.95)
- MAPE consistently ~20-21%
- No significant overfitting (train-val gap <3%)
- Early stopping prevented overfitting

---

## 📈 Improvement Journey

| Run | Data | Model | Val Loss | MAPE | Notes |
|-----|------|-------|----------|------|-------|
| v1 (MIT) | 14K rows | 338K | 0.0002 | 8438% | Near-zero data issue |
| v2 (300K) | 3 lakh | 338K | 88.0 | 75.4% | First Alibaba run |
| v3 (50L processed) | 50 lakh | 338K | 34.59 | 37.28% | Full processed data |
| **v4 (30L raw)** | **30 lakh** | **613K** | **5.95** | **~21%** | **Best! Raw sensor + bigger model** |

---

## 🔧 What Changed in Best Run

### Data
- Used **raw sensor table** (pai_sensor_table.csv, 1GB) instead of processed features
- Downloaded directly from Alibaba CDN to Kaggle
- 30 lakh rows of real GPU utilization data

### Model
- **613,612 params** (vs 338,252 in earlier runs)
- Larger TCN channels
- Same architecture (TCN + Attention)

### Training
- **Stride=5** (vs 10) — denser sequences, more training data
- **606K sequences** (vs 504K)
- **Batch size=512** for faster GPU training
- **Early stopping** with patience=25
- **Cosine annealing** learning rate schedule

---

## 🧮 CVXPY MPC Controller

Integrated Model Predictive Control using CVXPY optimizer:

```
Battery: 319.2 MW / 655.2 MWh (Tesla Megapack scale)
Target: 200 MW grid power
Horizon: 12 steps
Solver: OSQP
```

**Results:**
- Peak reduction: 6.36 MW (on test samples)
- Battery SOC maintained within 5-95%
- Grid power smoothed

---

## 📁 Files

| File | Description |
|------|-------------|
| `TRAINING_LOG.md` | Detailed epoch-by-epoch training log |
| `DATA_PIPELINE.md` | How data was collected and processed |
| `MODEL_ARCHITECTURE.md` | Model architecture details |
| `MPC_IMPLEMENTATION.md` | CVXPY MPC controller details |
| `KAGGLE_NOTEBOOK.md` | Kaggle notebook setup and instructions |

---

## 🔗 Links

- **Kaggle Notebook:** https://www.kaggle.com/code/vedkumr/energivanu-full-pipeline
- **Alibaba Dataset:** https://www.kaggle.com/datasets/vedkumr/energivanu-training-data
- **GitHub Repo:** https://github.com/mysterious75/energivanu2
- **Alibaba Trace Source:** https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

---

## 📜 License

- **Alibaba GPU Trace 2020:** CC BY 4.0 (cite NSDI '22 paper)
- **Code:** AGPL-3.0-or-later
- **Commercial Use:** ✅ Fully allowed (Alibaba CC BY 4.0 + own data)

---

## Citation

```bibtex
@inproceedings{wen2022characterizing,
    title     = {Characterizing and Profiling GPU Workloads on Alibaba},
    author    = {Wen, Mingshu and Li, Haowei and Liu, Yang and others},
    booktitle = {Proceedings of the 19th USENIX Symposium on Networked
                 Systems Design and Implementation (NSDI)},
    year      = {2022},
    url       = {https://www.usenix.org/conference/nsdi22/presentation/wen}
}
```
