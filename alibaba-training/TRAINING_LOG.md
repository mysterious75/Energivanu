# Training Log — Alibaba GPU Trace 2020

## Best Run: v4 (Raw Sensor Data + 613K Model)

### Environment
- **Platform:** Kaggle
- **GPU:** Tesla P100-PCIE-16GB (16GB VRAM)
- **CUDA:** 11.8 (via PyTorch cu118 for P100 compatibility)
- **PyTorch:** 2.7.1+cu118

### Data Loading
```
[0:00] Installing PyTorch cu118 + CVXPY...
[2:27] GPU: Tesla P100-PCIE-16GB, VRAM: 17.1 GB
[2:27] DOWNLOADING FULL ALIBABA GPU TRACE 2020
[2:28] Downloading pai_sensor_table (~388MB)
[2:56] sensor: 28s, rc=0 ✅
[2:56] Downloading pai_machine_metric (~198MB)
[3:14] metric: 18s, rc=0 ✅
[3:34] Extracting...
[3:36] pai_sensor_table: 1057MB, header added ✅
[3:36] pai_machine_metric: 437MB, header added ✅
```

### Feature Engineering
```
[3:36] PROCESSING FULL DATA → 15 FEATURES
[3:36] Loading pai_sensor_table.csv (ALL rows)...
[3:51] Loaded 3,033,232 rows in 15s
[3:51] ✅ pai_sensor_table.csv: (3033232, 16), gpu_util found
[3:52] Total rows: 3,033,232
[3:52] Features: (3033232, 15)
[3:52] Power: 14.0 - 140.0 MW
[3:54] Creating sequences (stride=5)...
[3:54] Sequences: (606639, 30, 15)
[3:55] Train: 515,643, Val: 90,996
```

### Model
```
Model: 613,612 params on cuda
Architecture: TCN (32→64→128) + Multi-Head Attention (8 heads)
```

### Training (200 epochs max, early stop patience=25)
```
[4:00] TRAINING — 200 epochs max (early stop patience=25)

Ep   1/200 | Train 7.6598 | Val 6.6770 | MAPE 21.43% | Best@1=6.6770 | 47.4s/ep | Total 47s
Ep   2/200 | Train 6.4692 | Val 6.3606 | MAPE 21.23% | Best@2=6.3606 | 47.4s/ep | Total 95s
Ep   3/200 | Train 6.3379 | Val 6.5668 | MAPE 25.84% | Best@2=6.3606 | 47.1s/ep | Total 142s
Ep   4/200 | Train 6.3117 | Val 6.2856 | MAPE 20.91% | Best@4=6.2856 | 47.1s/ep | Total 189s
Ep   5/200 | Train 6.3022 | Val 6.1353 | MAPE 21.75% | Best@5=6.1353 | 47.0s/ep | Total 236s
Ep  10/200 | Train 6.1582 | Val 6.3048 | MAPE 20.29% | Best@5=6.1353 | 47.5s/ep | Total 471s
Ep  20/200 | Train 6.0270 | Val 6.0400 | MAPE 20.30% | Best@19=6.0356 | 47.1s/ep | Total 942s
Ep  22/200 | Train      - | Val 5.9874 | MAPE     -  | Best@22=5.9874 |          |
Ep  30/200 | Train 5.9549 | Val 5.9924 | MAPE 20.66% | Best@22=5.9874 | 46.3s/ep | Total 1410s
Ep  40/200 | Train 5.8925 | Val 6.0534 | MAPE 21.69% | Best@22=5.9874 | 46.0s/ep | Total 1874s
Ep  41/200 | Train      - | Val 5.9477 | MAPE     -  | Best@41=5.9477 |          |
Ep  50/200 | Train 5.8309 | Val 6.0526 | MAPE 21.69% | Best@41=5.9477 | 45.9s/ep | Total 2334s
Ep  60/200 | Train 5.7691 | Val 6.0500 | MAPE 21.40% | Best@41=5.9477 | 45.9s/ep | Total 2793s
```

### Early Stopping
- Best at epoch 41: val_loss = 5.9477
- Patience counter started at epoch 41
- No improvement for 19 epochs (60-41)
- Would early stop at epoch 66 (41+25) if continued

### Key Metrics
```
Best Val Loss:     5.9477 (epoch 41)
Best MAPE:         ~20.3% (epoch 10, 20)
Final Train Loss:  5.7691 (epoch 60)
Final Val Loss:    6.0500 (epoch 60)
Overfitting Gap:   ~3% (train=5.77, val=6.05)
Training Speed:    ~46s/epoch
Total Time:        ~45 min
```
