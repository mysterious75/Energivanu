# ENERGIVANU — GitHub Repos & Implementations

## Time Series Forecasting Libraries

### 1. Time-Series-Library (THUML) — MOST IMPORTANT
**URL:** https://github.com/thuml/Time-Series-Library
**Stars:** High (Tsinghua University ML Lab)

Contains: Informer, Autoformer, PatchTST, TimesNet, iTransformer, FEDformer, and many more.

**Key features:**
- Unified training scripts for all models
- Zero-shot forecasting support for foundation models
- Custom model template available
- Scripts in `./scripts/` reproduce paper results

**Usage:**
```bash
# Train TimesNet on ETTh1
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# Zero-shot for foundation models
--task_name zero_shot_forecast --is_training 0
```

**Why important:** Single library to test ALL architectures on our data.

---

### 2. cure-lab/LTSF-Linear — DLinear/NLinear
**URL:** https://github.com/cure-lab/LTSF-Linear
**Stars:** High (famous paper)

Contains: DLinear, NLinear — simple linear models that beat Transformers.

**Key insight:** "Are Transformers Effective for Time Series Forecasting?"

**Why important:** We're already using DLinear/NLinear, but can compare with official implementation.

---

### 3. yuqinie98/PatchTST — PatchTST
**URL:** https://github.com/yuqinie98/PatchTST
**Stars:** High (ICLR 2023)

**Key features:**
- Patches time series into subseries-level tokens
- Channel-independent (each variate shares weights)
- Self-supervised pre-training via masked patch prediction
- 21% MSE reduction, 16.7% MAE reduction vs baselines

**Pre-training:**
```bash
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```

**Fine-tuning:**
```bash
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

**Why important:** Best Transformer variant for time series. Could replace our Transformer.

---

### 4. POWER-CAST — Energy Demand Forecasting
**URL:** https://github.com/Chan-dre-yi/POWER-CAST
**Stars:** 9

**Key features:**
- Temporal Fusion Transformer (TFT) for energy demand
- Direct relevance to power forecasting

**Why important:** Already applied TFT to energy domain.

---

### 5. pytorch-forecasting — TFT Implementation
**URL:** https://github.com/jdb78/pytorch-forecasting
**Stars:** High

**Key features:**
- Temporal Fusion Transformer (TFT) in PyTorch
- Variable selection networks (interpretable)
- Learning rate finder
- Built-in data handling with PyTorch Lightning

**Why important:** Production-ready TFT implementation.

---

## Foundation Model Repos

### 6. Amazon Chronos-2 — BEST FOR ZERO-SHOT
**URL:** https://github.com/amazon-science/chronos-forecasting
**Stars:** High (Amazon Science)

**Architecture:**
- Chronos-T5: T5 language model for time series tokens
- Chronos-Bolt: Patch-based, 250x faster, 20x more memory efficient
- Chronos-2: Zero-shot for univariate, multivariate, covariates

**Pre-trained weights (HuggingFace):**
| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Chronos-2 | 120M | `amazon/chronos-2` |
| Chronos-Bolt-base | 205M | `amazon/chronos-bolt-base` |
| Chronos-T5-base | 200M | `amazon/chronos-t5-base` |
| Chronos-T5-large | 710M | `amazon/chronos-t5-large` |

**Kaggle T4 compatible:** Yes (up to 205M comfortably)

**Why important:** Best zero-shot performance. Can predict without training.

---

### 7. Google TimesFM 2.5 — BEST FOR FINE-TUNING
**URL:** https://github.com/google-research/timesfm
**Stars:** High (Google Research)

**Architecture:**
- Decoder-only foundation model
- 200M parameters, 16k context length
- PyTorch and JAX backends
- Pre-trained on 100 billion time series data points

**Pre-trained weights:**
- `google/timesfm-2.5-200m-pytorch`

**Fine-tuning:**
- Example in `timesfm-forecasting/examples/finetuning/`
- Use LoRA (PEFT) for memory-efficient fine-tuning

**Kaggle T4 compatible:** Yes (200M with LoRA)

**Why important:** Google's model, pre-trained on massive data, LoRA fine-tuning supported.

---

### 8. Salesforce Moirai — UNIVERSAL FORECASTING
**URL:** https://github.com/SalesforceAIResearch/uni2ts
**Stars:** High (Salesforce AI)

**Architecture:**
- Masked Encoder-based Universal Transformer
- Patch-based tokenization (configurable patch sizes)
- Any number of variates, any frequency, any prediction length
- Trained on LOTSA (Large-scale Open Time Series Archive)

**Pre-trained weights:**
| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Moirai 2.0-R | ~14M | `Salesforce/moirai-2.0-R-small` |
| Moirai 1.1-R | ~91M | `Salesforce/moirai-1.1-R-base` |
| Moirai 1.1-R | ~311M | `Salesforce/moirai-1.1-R-large` |

**Fine-tuning:**
```bash
python -m cli.train -cp conf/finetune
```

**Kaggle T4 compatible:** Yes (small=14M, base=91M both fit)

**Why important:** Universal model, handles multivariate natively.

---

### 9. Lag-Llama — PROBABILISTIC FORECASTING
**URL:** https://github.com/time-series-foundation-models/Lag-Llama
**Stars:** Moderate

**Architecture:**
- Decoder-only Transformer (LLaMA-style)
- Causal attention with Rotary Position Embeddings (RoPE)
- Lagged time series values as input tokens
- Outputs probability distributions

**Pre-trained weights:**
- `time-series-foundation-models/Lag-Llama` on HuggingFace

**Fine-tuning:**
- Colab demo available
- Key hyperparameters: context length (32-1024), learning rate (10^-4 to 10^-2)
- Early stopping with patience=50

**Kaggle T4 compatible:** Yes (4-8GB typical)

**Why important:** Probabilistic predictions with uncertainty.

---

## Battery & Energy Management Repos

### 10. battery-capacity-fade-prediction
**URL:** https://github.com/connectashish028/battery-capacity-fade-prediction
**Stars:** 2

**Key features:**
- ML for battery capacity prediction over time
- Multiple model evaluation

**Why important:** Battery degradation modeling for Tesla Megapack.

---

### 11. EV-Smart-Management-System
**URL:** https://github.com/siddharth23k/EV-Smart-Management-System
**Stars:** 1

**Key features:**
- AI-powered braking intention prediction
- Battery State-of-Charge estimation
- Regenerative braking control optimization

**Why important:** SoC estimation techniques applicable to grid-scale batteries.

---

### 12. Smart-battery-management (Deep RL)
**URL:** https://github.com/albin-shajan-2004/Smart-battery-management-A7-
**Stars:** 0

**Key features:**
- Deep Learning & RL for SoC, SoH, RUL Estimation
- LSTM and GRU networks
- Hyperparameter optimization

**Why important:** RL approach for battery management.

---

## Other Useful Repos

### 13. TSDB — Time Series Dataset Toolbox
**URL:** https://github.com/WenjieDu/TSDB
**Stars:** 236

**Key features:**
- 173 public time series datasets
- Single-line loading

**Why important:** Benchmark datasets for testing models.

---

### 14. LTSF-Models-Collection
**URL:** https://github.com/Helios-17s/LTSF-Models-Collection
**Stars:** 0 (new)

**Key features:**
- Autoformer, DLinear, Informer, Transformer, PatchTST, Reformer, SegRNN, STGCN

**Why important:** All models in one place for comparison.

---

## Recommended Implementation Order

| Priority | Repo/Model | Why |
|----------|-----------|-----|
| P0 | Chronos-2 (amazon/chronos-forecasting) | Zero-shot, no training needed |
| P0 | TimesFM 2.5 (google-research/timesfm) | Fine-tune with LoRA |
| P1 | PatchTST (yuqinie98/PatchTST) | Best Transformer variant |
| P1 | Time-Series-Library (thuml/Time-Series-Library) | All models unified |
| P2 | Moirai (SalesforceAIResearch/uni2ts) | Universal, multivariate |
| P2 | Lag-Llama | Probabilistic predictions |
| P3 | pytorch-forecasting | TFT implementation |
| P3 | POWER-CAST | Energy domain specific |

---

## Next Steps

1. Clone Chronos-2 and test zero-shot on our data
2. Clone TimesFM and test fine-tuning with LoRA
3. Clone Time-Series-Library for unified model comparison
4. Test PatchTST for potential Transformer replacement
5. Implement battery management algorithms from battery repos
