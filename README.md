# ENERGIVANU

**AI-Powered Energy Management for GPU Supercomputers**

Predicts power demand spikes 10 minutes ahead and signals Tesla Megapack batteries before grid instability.

## Quick Start

```bash
# 1. Generate project
python setup.py

# 2. Upload energivanu_colab.ipynb to Google Colab

# 3. Run all cells in Colab (training happens on free T4 GPU)
```

## Architecture

```
GPU Telemetry + Weather + Grid → Features → Transformer → Power + Signal → Battery
```

## Project Name: ENERGIVANU
