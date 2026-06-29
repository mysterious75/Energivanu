# Kaggle Notebook Setup

## Notebook URL
https://www.kaggle.com/code/vedkumr/energivanu-full-pipeline

## Setup Steps
1. Go to Kaggle → New Notebook
2. Enable **GPU** (P100 or T4)
3. Enable **Internet**
4. Add datasets:
   - `vedkumr/energivanu-training-data` (processed features)
   - `vedkumr/mit-supercloud-real2` (validation data)
5. Copy `energivanu-full-pipeline.py` code
6. Run All

## P100 GPU Compatibility Fix
Kaggle's default PyTorch (2.10+cu128) doesn't support P100 (sm_60).
Fix: Install PyTorch cu118 at the start of the notebook:
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
```

## Data Download (Alternative)
If dataset upload fails, download directly in notebook:
```python
os.system("curl -sL 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz' -o sensor.tar.gz")
os.system("tar xzf sensor.tar.gz")
# Add header
os.system("cat pai_sensor_table.header pai_sensor_table.csv > tmp.csv && mv tmp.csv pai_sensor_table.csv")
```

## Expected Runtime
| Step | Time |
|------|------|
| PyTorch cu118 install | ~2 min |
| Data download | ~1 min |
| Data processing | ~30 sec |
| 200 epochs training | ~45 min |
| CVXPY MPC test | ~30 sec |
| **Total** | **~50 min** |

## GPU Sessions
- Kaggle allows 2 GPU sessions per week (39 hrs/week)
- P100 and T4 both work with cu118
- If GPU quota exhausted, use CPU (will be much slower)

## Outputs
- `best_model.pt` — trained model checkpoint
- `results.json` — training metrics
- `energivanu-full-pipeline.log` — full training log
