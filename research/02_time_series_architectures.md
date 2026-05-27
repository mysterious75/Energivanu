# ENERGIVANU - Time Series Forecasting Architecture Research

**Date:** 2026-05-27
**Purpose:** Evaluate alternative architectures to replace the current ColossusTransformer
**Current Problem:** Overfitting on synthetic data (val loss increases from epoch 1, DirAcc stuck at 50%)
**Dataset:** 518K samples, 34 features, 60-step lookback, 60-step horizon (5-second intervals)

---

## Table of Contents

1. [Why Transformers Overfit on Time Series](#1-why-transformers-overfit-on-time-series)
2. [DLinear / NLinear](#2-dlinear--nlinear)
3. [PatchTST](#3-patchtst)
4. [Informer](#4-informer)
5. [Autoformer](#5-autoformer)
6. [iTransformer](#6-itransformer)
7. [TimesNet](#7-timesnet)
8. [TSMixer](#8-tsmixer)
9. [FEDformer](#9-fedformer)
10. [State-of-the-Art 2025-2026: Foundation Models & SSMs](#10-state-of-the-art-2025-2026-foundation-models--ssms)
11. [Architecture Comparison Matrix](#11-architecture-comparison-matrix)
12. [Recommendations for ENERGIVANU](#12-recommendations-for-energivanu)

---

## 1. Why Transformers Overfit on Time Series

### The Core Problem

Transformers were designed for NLP where tokens are discrete, semantically rich, and permutable. Time series data has fundamentally different properties:

**1. Permutation-Invariant Self-Attention vs. Ordered Temporal Data**
- Self-attention computes Q*K^T to find relationships between tokens. This is permutation-invariant by design.
- Time series data is inherently ordered -- the position of a value matters critically.
- Positional encoding (sinusoidal or learned) attempts to inject order, but research shows it is insufficient to fully preserve temporal structure (Zeng et al., 2023).

**2. Data Hunger vs. Dataset Size**
- Transformers have O(L^2) attention parameters per layer. For a 60-step lookback with 34 features, this creates many free parameters.
- 518K samples sounds large, but for a Transformer with millions of parameters, this is insufficient for generalization.
- The model memorizes training patterns rather than learning transferable temporal dynamics.

**3. Synthetic Data Smoothness**
- Our synthetic data has smooth sinusoidal patterns (8-hour and 24-hour cycles) with predictable spikes.
- Transformers excel at capturing complex, irregular patterns. On smooth data, they overfit to noise.
- Simple linear models naturally capture smooth trends without memorization.

**4. Non-Stationarity**
- Real time series have shifting distributions (mean, variance change over time).
- Transformers process raw values without inherent normalization, leading to distribution-specific memorization.
- RevIN (Kim et al., 2022) was proposed to address this, but it adds complexity.

**5. Benchmark Overfitting (Literature Problem)**
- Many Transformer time series papers tuned heavily on ETT, Weather, Electricity benchmarks.
- When simple linear models (DLinear) were tested, they beat most Transformers on these same benchmarks.
- This suggests Transformers were overfitting to benchmark artifacts, not learning general temporal patterns.

### Known Failure Modes

| Failure Mode | Symptom | Our Case |
|---|---|---|
| **Early stopping failure** | Val loss increases from epoch 1 | Confirmed: best MAE at epoch 1 |
| **Attention collapse** | All attention weights become uniform | Likely with smooth synthetic data |
| **Positional encoding saturation** | PE dominates over actual values | Possible with 60-step lookback |
| **Feature interaction noise** | Cross-feature attention adds noise | 34 features, many redundant |
| **Horizon degradation** | Accuracy drops sharply beyond short horizon | 60-step horizon is challenging |

### Why DLinear Already Helps (Our Current Implementation)

Our existing DLinear (`src/models/dlinear.py`) uses:
- Moving average decomposition (AvgPool1d kernel=25)
- Separate linear layers for trend and seasonal
- Channel-independent processing (no cross-feature mixing)
- Only ~12K parameters vs ~1M for Transformer

This explains why DLinear was set as `model_type = "dlinear"` in config -- it was likely found to be more stable than the full Transformer.

---

## 2. DLinear / NLinear

**Paper:** "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., AAAI 2023)
**Code:** https://github.com/cure-lab/LTSF-Linear

### How It Works

**NLinear (Normalization + Linear):**
```
Input (B, L, F)
  → Subtract last value per series: x' = x - x[:, -1:, :]
  → Single linear layer: Linear(L → H)
  → Add back last value: output = Linear(x') + x[:, -1:, :]
```
- The subtraction trick removes non-stationarity (each series is zero-centered at prediction start).
- One learnable weight matrix of shape (H, L) per feature.

**DLinear (Decomposition + Linear):**
```
Input (B, L, F)
  → Decompose: Trend = AvgPool(x), Seasonal = x - Trend
  → Two linear layers: Linear_trend(L → H), Linear_seasonal(L → H)
  → Sum: output = Linear_trend(Trend) + Linear_seasonal(Seasonal)
```
- Moving average kernel extracts the trend component.
- Each component has its own linear projection.
- Total parameters: 2 * (H * L) per feature.

### Key Results

- Outperformed Informer, Autoformer, FEDformer, Pyraformer, LogTrans on 9 benchmarks.
- On ETTh1 (96-step): DLinear MSE=0.080 vs Informer MSE=0.865 (10x better).
- On Weather (96-step): DLinear MSE=0.003 vs Autoformer MSE=0.014.
- The paper concluded: "The success of Transformers in time series is not due to the attention mechanism."

### Pros for ENERGIVANU

- **Extremely simple**: ~12K parameters, impossible to overfit on 518K samples.
- **Fast inference**: Single matrix multiplication, well under 100ms.
- **Already implemented**: We have DLinear in `src/models/dlinear.py`.
- **Channel-independent**: Each of 34 features processed separately, avoiding cross-feature noise.
- **Robust to synthetic data**: Linear models naturally capture smooth sinusoidal patterns.

### Cons for ENERGIVANU

- **No cross-feature interaction**: Cannot learn that GPU temp correlates with power.
- **No nonlinear patterns**: Cannot capture threshold effects (e.g., thermal throttling).
- **Limited expressiveness**: If real data has complex patterns, linear model will underfit.
- **Direction accuracy**: Linear model cannot predict up/down direction well (still ~50%).
- **Horizon degradation**: 60-step horizon is long; linear extrapolation becomes unreliable.

### Implementation Complexity: LOW (already done)

### Expected Performance vs Current Transformer
- Likely **better** on synthetic data (less overfitting).
- Likely **worse** on real data with complex patterns.
- DirAcc will remain ~50% (no nonlinear direction modeling).

---

## 3. PatchTST

**Paper:** "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (Nie et al., ICLR 2023)
**Code:** https://github.com/yuqinie98/PatchTST

### How It Works

PatchTST makes two key modifications to the vanilla Transformer:

**1. Patching (inspired by Vision Transformer):**
```
Input: (B, L, F) where L=lookback, F=features
  → Divide into patches: (B, N, P*F) where N=L/P patches, P=patch_size
  → Linear projection: (B, N, d_model)
  → Standard Transformer encoder
```
- Each patch contains P consecutive timesteps, preserving local temporal structure.
- Reduces token count by factor P (e.g., 60 steps / 10 patch_size = 6 tokens).
- Attention operates on patches, not individual time steps.

**2. Channel Independence:**
```
Each feature f in [1..F]:
  → Patch embedding: (B, N, P) → (B, N, d_model)
  → Transformer encoder (shared weights across features)
  → Prediction head: (B, N, d_model) → (B, H)
```
- Each feature is processed independently through the same Transformer.
- Parameters are shared across features, but no cross-feature attention.
- Avoids the "curse of dimensionality" with many features.

**3. Self-Supervised Pre-training (optional):**
- Masked patch prediction: randomly mask 40% of patches, reconstruct.
- Enables transfer learning across datasets.
- Can pre-train on large unlabeled time series, fine-tune on ENERGIVANU.

### Key Results

- State-of-the-art on ETTh1, ETTh2, Weather, Electricity, Traffic at time of publication.
- 96-step forecasting: PatchTST MSE=0.070 vs DLinear MSE=0.080 on ETTh1.
- Pre-trained PatchTST transfers well across domains.

### Pros for ENERGIVANU

- **Patching reduces tokens**: 60-step / 10-patch = 6 tokens. Very efficient attention.
- **Local semantic information**: Patches preserve short-term patterns (spikes, ramps).
- **Channel independence**: Handles 34 features without cross-feature overfitting.
- **Pre-training potential**: Can pre-train on synthetic data, fine-tune on real data later.
- **Well-tested architecture**: ICLR 2023, extensive benchmarks.

### Cons for ENERGIVANU

- **Still a Transformer**: Retains quadratic attention within the reduced token space.
- **No cross-feature interaction**: Same limitation as DLinear for correlated features.
- **More complex than DLinear**: ~100K-500K parameters depending on d_model.
- **Patch size sensitivity**: Wrong patch size can hurt performance.
- **Overfitting risk**: With 34 features * 6 tokens, still has many parameters for 518K samples.

### Implementation Complexity: MEDIUM
- Need to implement patching, channel-independent forward pass.
- Can reuse existing PatchEmbed from our Transformer.
- ~200-400 lines of new code.

### Expected Performance vs Current Transformer
- **Better** generalization due to patching and channel independence.
- **Better** long-horizon performance (patches capture local patterns).
- **Similar** DirAcc (~50%) unless augmented with direction-specific loss.
- **Lower** overfitting risk than vanilla Transformer.

---

## 4. Informer

**Paper:** "Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (Zhou et al., AAAI 2021 Best Paper)
**Code:** https://github.com/zhouhaoyi/Informer2020

### How It Works

Informer addresses the O(L^2) complexity of standard Transformers with three innovations:

**1. ProbSparse Self-Attention:**
```
Standard attention: A = softmax(QK^T / sqrt(d)) * V  → O(L^2)
ProbSparse: Select top-u queries based on KL-divergence from uniform
  → Only u << L queries attend to all keys
  → Complexity: O(L * log(L))
```
- Intuition: Most query-key interactions produce near-uniform attention weights (uninformative).
- Only "active" queries (those with peaked attention distributions) contribute meaningfully.
- Uses KL-divergence to score query importance, selects top-u.

**2. Self-Attention Distilling:**
```
Layer 1: (B, L, d_model) → attention → (B, L, d_model)
Layer 2: (B, L/2, d_model) → halve by conv → attention
Layer 3: (B, L/4, d_model) → halve by conv → attention
```
- Progressively halves sequence length across layers.
- Focuses computation on the most important features.
- Reduces memory footprint for long sequences.

**3. Generative Style Decoder:**
```
Instead of autoregressive: y_t = f(y_{t-1}, ..., y_{t-k}, x)
Use: y_all = f(x) in one forward pass
```
- Predicts all H future steps simultaneously.
- Much faster inference than step-by-step decoding.

### Key Results

- AAAI 2021 Best Paper.
- On ETTh1 (24-step): Informer MSE=0.098 vs LSTM MSE=0.154.
- Significant speedup over vanilla Transformer for long sequences.

### Pros for ENERGIVANU

- **Efficient for long sequences**: O(L log L) is better than O(L^2).
- **Generative decoder**: Single-pass prediction for all 60 future steps.
- **Established architecture**: Widely cited, well-understood.
- **Distilling mechanism**: Reduces redundant features across layers.

### Cons for ENERGIVANU

- **ProbSparse is approximate**: May miss important attention patterns.
- **Complex implementation**: ProbSparse requires custom attention kernels.
- **Superseded by newer models**: PatchTST, iTransformer outperform it.
- **Still a Transformer**: Retains fundamental issues with temporal data.
- **Our sequence is short (60)**: O(L log L) vs O(L^2) difference is minimal at L=60.

### Implementation Complexity: HIGH
- Custom ProbSparse attention mechanism.
- Self-attention distilling layers.
- Generative decoder design.
- ~500+ lines of new code.

### Expected Performance vs Current Transformer
- **Similar** or slightly better due to sparsity.
- **Faster** inference for longer sequences (not significant at L=60).
- **Same overfitting risk**: ProbSparse doesn't address the fundamental issue.

---

## 5. Autoformer

**Paper:** "Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (Wu et al., NeurIPS 2021)
**Code:** https://github.com/thuml/Autoformer

### How It Works

Autoformer replaces self-attention with auto-correlation and embeds decomposition into the architecture:

**1. Auto-Correlation Mechanism:**
```
Standard attention: A = softmax(QK^T) * V  (point-wise)
Auto-correlation: A = PeriodBasedAggregate(R(q,k), V)  (sub-series level)
```
- Uses the autocorrelation function to discover periodic dependencies.
- Instead of point-wise dot products, it aggregates entire periodic sub-series.
- Implemented via FFT: R(q,k) = IFFT(FFT(q) * conj(FFT(k)))
- Time delay aggregation rolls the value series to align with discovered periods.

**2. Deep Decomposition Architecture:**
```
Input → [Auto-Correlation Layer → Series Decomposition → Feed-Forward] * N layers
         Trend + Seasonal extraction at EVERY layer
```
- Traditional approach: decompose once as preprocessing.
- Autoformer: decompose progressively at each layer.
- Each layer refines the trend-seasonal separation.
- Final prediction: sum of trend and seasonal predictions.

**3. Series Decomposition Block:**
```
Trend = AvgPool(x, kernel=k)
Seasonal = x - Trend
```
- Moving average kernel extracts trend.
- Residual is the seasonal component.

### Key Results

- 38% relative improvement across 6 benchmarks.
- State-of-the-art on ETTh1, ETTh2, Weather, Electricity, Traffic at publication.
- NeurIPS 2021.

### Pros for ENERGIVANU

- **Progressive decomposition**: Handles complex trend-seasonal patterns better than single decomposition.
- **Sub-series attention**: Captures periodic patterns (8-hour, 24-hour cycles) naturally.
- **FFT-based**: Efficient implementation, O(L log L).
- **Domain-relevant**: Energy data has strong periodic patterns (daily, weekly).

### Cons for ENERGIVANU

- **Complex architecture**: Auto-correlation + decomposition is hard to implement correctly.
- **Period detection sensitivity**: FFT-based period discovery may fail on noisy synthetic data.
- **Superseded**: Newer models (PatchTST, iTransformer) outperform it.
- **Still has Transformer limitations**: Positional encoding, embedding issues remain.
- **Overfitting risk**: Complex architecture on synthetic data.

### Implementation Complexity: HIGH
- Custom auto-correlation mechanism with FFT.
- Series decomposition blocks.
- Modified encoder-decoder architecture.
- ~600+ lines of new code.

### Expected Performance vs Current Transformer
- **Better** if data has strong periodic patterns (our synthetic data does).
- **Similar** overfitting risk due to complexity.
- **Better** trend-seasonal separation than vanilla Transformer.

---

## 6. iTransformer

**Paper:** "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (Liu et al., ICLR 2024)
**Code:** https://github.com/thuml/iTransformer

### How It Works

iTransformer makes a radical but simple change: **invert the dimensions** of the Transformer input.

**Standard Transformer:**
```
Input: (B, L, F) → each token = one timestep with all features
Attention: across time steps (L tokens)
FFN: applied to each timestep token
```

**iTransformer (Inverted):**
```
Input: (B, L, F) → transpose → (B, F, L)
Each token = one feature's entire time series (L values)
Attention: across features (F tokens)
FFN: applied to each feature token (encodes temporal patterns)
```

**Key Insight:**
- In standard Transformer, attention across time steps produces "meaningless attention maps" because temporal patterns are better captured by FFN.
- In iTransformer, attention across features captures multivariate correlations (e.g., GPU temp correlates with power).
- FFN on individual feature tokens captures nonlinear temporal patterns.

**Architecture:**
```
Input (B, L, F)
  → Instance normalization (per feature)
  → Embedding: each feature's L timesteps → d_model vector
  → Transformer encoder: attention across F feature tokens
  → FFN: nonlinear temporal encoding per feature
  → Projection: d_model → H (forecast horizon)
  → Reverse normalization
```

### Key Results

- State-of-the-art on ETTh1, ETTh2, Weather, Electricity, Traffic (2024).
- Better generalization across different numbers of variates.
- Better utilization of arbitrary lookback windows.
- ICLR 2024.

### Pros for ENERGIVANU

- **Captures cross-feature correlations**: GPU temp, load, power are correlated. iTransformer learns this.
- **No architectural modifications**: Standard Transformer components, just inverted.
- **Better with many features**: 34 features become 34 tokens (manageable).
- **Instance normalization**: Handles non-stationarity naturally.
- **Proven SOTA**: Extensive benchmarks on energy, weather, traffic data.

### Cons for ENERGIVANU

- **Attention across 34 features**: 34 tokens is very few for attention to be meaningful.
- **Still a Transformer**: Retains positional encoding issues (but less relevant since attention is across features).
- **FFN must encode all temporal patterns**: Relies heavily on FFN capacity.
- **More complex than DLinear**: ~200K-500K parameters.
- **Overfitting risk**: With 34 feature tokens, attention may memorize feature correlations from training data.

### Implementation Complexity: MEDIUM
- Invert input dimensions (transpose).
- Instance normalization layer.
- Standard Transformer encoder (can reuse existing).
- ~150-300 lines of code modification.

### Expected Performance vs Current Transformer
- **Significantly better** due to meaningful attention (across features, not time).
- **Better** cross-feature learning (GPU temp → power prediction).
- **Lower** overfitting risk than standard Transformer.
- **Better** DirAcc if direction correlates with feature interactions.

---

## 7. TimesNet

**Paper:** "Temporal 2D-Variation Modeling for General Time Series Analysis" (Wu et al., ICLR 2023)
**Code:** https://github.com/thuml/TimesNet

### How It Works

TimesNet transforms 1D time series into 2D representations based on detected periods:

**1. Period Detection via FFT:**
```
Input: 1D series x of length L
  → FFT: X = FFT(x)
  → Amplitude spectrum: A = |X|
  → Top-k periods: p_1, p_2, ..., p_k (highest amplitude frequencies)
```

**2. 2D Reshape:**
```
For each period p:
  → Reshape 1D series (L,) into 2D tensor (p, L/p)
  → Rows = one full period cycle
  → Columns = same position across cycles
  → Intra-period variation: patterns within one cycle (columns)
  → Inter-period variation: patterns across cycles (rows)
```

**3. TimesBlock:**
```
For each detected period p:
  → Reshape to 2D: (B, L/p, p)
  → Apply 2D convolution (inception-style kernel)
  → Reshape back to 1D: (B, L)
  → Weighted sum across all periods
```

**4. Multi-Periodicity:**
- Multiple periods are detected (e.g., 8-hour, 24-hour, weekly).
- Each period produces a separate 2D representation.
- Results are aggregated with learned weights.

### Key Results

- State-of-the-art across 5 tasks: forecasting, classification, imputation, anomaly detection, short-term forecasting.
- ICLR 2023.
- Unified framework for multiple time series tasks.

### Pros for ENERGIVANU

- **Multi-periodicity detection**: Our data has 8-hour and 24-hour cycles. TimesNet can discover these automatically.
- **2D convolution**: Captures both intra-period and inter-period patterns.
- **Unified framework**: Can be used for forecasting + anomaly detection (useful for spike detection).
- **Parameter-efficient**: Inception-style kernels are lightweight.
- **Domain-relevant**: Energy data has strong periodic patterns.

### Cons for ENERGIVANU

- **FFT period detection sensitivity**: May detect spurious periods on noisy synthetic data.
- **2D convolution overhead**: Reshaping adds computational cost.
- **Complex implementation**: Period detection, 2D reshape, inception blocks.
- **Not primarily a Transformer**: Uses CNN, not attention. Different paradigm.
- **Overfitting risk**: Multiple period branches increase parameter count.

### Implementation Complexity: HIGH
- FFT-based period detection.
- 2D reshape and inception blocks.
- Multi-period aggregation.
- ~400-600 lines of new code.

### Expected Performance vs Current Transformer
- **Better** if periodic patterns are dominant (our synthetic data: yes).
- **Better** at capturing multi-scale temporal patterns.
- **Similar** overfitting risk (different architecture, similar parameter count).
- **Better** for anomaly detection component of ENERGIVANU.

---

## 8. TSMixer

**Paper:** "TSMixer: An All-MLP Architecture for Time Series Forecasting" (Chen et al., Google Research, KDD 2023)
**arXiv:** arXiv:2303.06053
**Code:** https://github.com/google-research/google-research/tree/master/tsmixer

### How It Works

TSMixer replaces attention with alternating MLP blocks:

**Architecture:**
```
Input: (B, L, F)

For each mixer block:
  → Time-Mixing MLP:
      LayerNorm → Linear(L → L) → GELU → Linear(L → L) → Residual
      (Mixes information across time steps for each feature)

  → Feature-Mixing MLP:
      LayerNorm → Linear(F → F) → GELU → Linear(F → F) → Residual
      (Mixes information across features at each time step)

Output: Linear projection to horizon
```

**Key Design:**
- Time-mixing: captures temporal dependencies (like a temporal attention, but simpler).
- Feature-mixing: captures cross-variate correlations (like channel attention).
- Alternating blocks: progressively refine representations.
- Residual connections + LayerNorm: stable training.

**Variants:**
- **TSMixer**: Standard version with time-mixing + feature-mixing.
- **TSMixer-Ext**: Extended with temporal projections for different input/output lengths.
- **TSMixer-Reconcile**: For hierarchical time series with reconciliation.

### Key Results

- Matched or outperformed Transformer-based models (FEDformer, Autoformer) on benchmarks.
- Significantly simpler and more efficient than Transformers.
- KDD 2023, Google Research.

### Pros for ENERGIVANU

- **Extremely simple**: Pure MLP, no attention, no FFT, no decomposition.
- **Low overfitting risk**: MLPs have fewer parameters than Transformers for same capacity.
- **Captures cross-feature interactions**: Feature-mixing learns GPU temp → power correlation.
- **Fast inference**: MLP is just matrix multiplication.
- **Easy to implement**: Standard PyTorch layers only.
- **Channel-mixing addresses DLinear's weakness**: DLinear is channel-independent; TSMixer mixes channels.
- **Well-suited for our data**: 34 features * 60 timesteps is a natural matrix shape.

### Cons for ENERGIVANU

- **No explicit temporal structure**: MLP doesn't know about ordering (same issue as Transformer).
- **No periodicity awareness**: Cannot discover 8-hour/24-hour cycles without explicit encoding.
- **Linear mixing**: Time-mixing MLP is linear across time, may miss nonlinear temporal patterns.
- **Less studied than Transformers**: Fewer papers, less community support.
- **Feature-mixing on 34 features**: 34x34 mixing matrix has 1156 parameters per layer.

### Implementation Complexity: LOW-MEDIUM
- Standard MLP blocks with LayerNorm and residuals.
- ~100-200 lines of new code.
- No custom attention or FFT required.

### Expected Performance vs Current Transformer
- **Better** generalization (simpler architecture, less overfitting).
- **Better** cross-feature learning (feature-mixing vs no mixing in DLinear).
- **Similar** or better MAE on synthetic data.
- **Better** DirAcc if direction correlates with feature interactions.
- **Faster** inference.

---

## 9. FEDformer

**Paper:** "Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting" (Zhou et al., ICML 2022)
**Code:** https://github.com/MAZiqing/FEDformer

### How It Works

FEDformer combines frequency-domain attention with trend-seasonal decomposition:

**1. Frequency-Enhanced Attention:**
```
Standard attention: A = softmax(QK^T / sqrt(d)) * V  → O(L^2)
FEDformer: A = IFFT(FFT(Q) * FFT(K)) * V  → O(L)
```
- Operates in the frequency domain using Fourier transforms.
- Exploits sparsity: most time series have sparse frequency representations.
- Linear complexity O(L) instead of quadratic O(L^2).

**2. Mixture of Experts in Frequency Domain:**
```
For each frequency component:
  → Expert 1: Fourier basis
  → Expert 2: Wavelet basis
  → Gating network selects expert weights
```
- Multiple frequency bases capture different types of patterns.
- Gating network adaptively selects the best representation.

**3. Decomposition Architecture:**
```
Input → [Frequency Attention → Series Decomposition → Feed-Forward] * N layers
         Trend + Seasonal separation at each layer
```
- Similar to Autoformer's progressive decomposition.
- Separates trend and seasonal components at each layer.

### Key Results

- 14.8% error reduction for multivariate, 22.6% for univariate vs SOTA.
- Linear complexity O(L).
- ICML 2022.

### Pros for ENERGIVANU

- **Frequency-domain processing**: Natural fit for periodic energy data.
- **Linear complexity**: Efficient for long sequences.
- **Mixture of experts**: Handles multiple periodicities (8h, 24h, weekly).
- **Decomposition**: Separates trend from seasonal patterns.
- **Domain-relevant**: Designed for energy, weather, traffic data.

### Cons for ENERGIVANU

- **Complex implementation**: Frequency attention + mixture of experts + decomposition.
- **FFT overhead**: Transforms add computational cost.
- **Superseded by newer models**: PatchTST, iTransformer outperform it.
- **Still a Transformer**: Retains fundamental issues with temporal data.
- **Overfitting risk**: Complex architecture on synthetic data.

### Implementation Complexity: HIGH
- Frequency-domain attention mechanism.
- Mixture of experts with Fourier and wavelet bases.
- Series decomposition blocks.
- ~600+ lines of new code.

### Expected Performance vs Current Transformer
- **Better** for periodic patterns (frequency domain).
- **Similar** overfitting risk (complex architecture).
- **Better** trend-seasonal separation.
- **Slower** inference due to FFT overhead.

---

## 10. State-of-the-Art 2025-2026: Foundation Models & SSMs

### 10.1 Time Series Foundation Models

**Chronos (Amazon, 2024)**
- Architecture: T5 (language model) adapted for time series.
- Tokenization: Numerical values are quantized into discrete tokens.
- Pre-training: Large-scale time series corpus.
- Zero-shot: Can forecast without task-specific training.
- Relevance to ENERGIVANU: Could use pre-trained Chronos for zero-shot forecasting, but may overfit to non-energy domains.

**TimesFM (Google, 2024)**
- Architecture: Decoder-only Transformer, ~200M parameters.
- Pre-training: Large-scale time series corpus.
- Zero-shot: Handles diverse domains without fine-tuning.
- Relevance to ENERGIVANU: Large model for our 518K samples; risk of overfitting during fine-tuning.

**MOIRAI (Salesforce, 2024)**
- Architecture: Masked encoder-based universal Transformer.
- Features: Variable-length I/O, multiple frequencies, any-variate attention.
- Pre-training: LOTSA (Large-scale Open Time Series Archive).
- Relevance to ENERGIVANU: Most flexible foundation model; could handle our 34 features.

**TimeGPT (Nixtla, 2024)**
- Architecture: Proprietary, API-based.
- Features: Zero-shot forecasting via API.
- Relevance to ENERGIVANU: Cannot run locally; API dependency.

### 10.2 State Space Models (SSMs)

**Mamba (Gu & Dao, 2023)**
- Architecture: Selective state space model.
- Complexity: O(L) linear complexity.
- Key innovation: Input-dependent selection mechanism.
- Relevance to ENERGIVANU: Efficient for long sequences; may capture temporal dynamics better than attention.

**S-Mamba (2024)**
- Architecture: Mamba adapted for time series forecasting.
- Features: Linear complexity, recurrent formulation.
- Benchmarks: Competitive with Transformer-based models.
- Relevance to ENERGIVANU: Could replace Transformer with linear-complexity SSM.

**TimeMixer (2024)**
- Architecture: Multi-scale mixing with SSM components.
- Features: Captures patterns at different temporal scales.
- Relevance to ENERGIVANU: Multi-scale approach could handle 8h and 24h cycles.

### 10.3 Hybrid Approaches

**Time-LLM (2024)**
- Architecture: Reprograms LLM (e.g., LLaMA) for time series.
- Features: Uses language model knowledge for time series.
- Relevance to ENERGIVANU: Overkill for our use case; high computational cost.

**Timer (2024)**
- Architecture: Generative pre-trained Transformer for time series.
- Features: Autoregressive generation of time series values.
- Relevance to ENERGIVANU: Could be fine-tuned for energy forecasting.

### 10.4 Recommendations for ENERGIVANU

Given our constraints (518K samples, 34 features, 60-step lookback, Kaggle/Colab):

1. **Foundation models are overkill**: 200M+ parameters for 518K samples = overfitting.
2. **SSMs (Mamba) are promising**: Linear complexity, but implementation is complex.
3. **Best bet**: Simple architectures (DLinear, TSMixer) with proper regularization.

---

## 11. Architecture Comparison Matrix

| Architecture | Complexity | Params | Overfitting Risk | Cross-Feature | Periodicity | DirAcc Potential | Implementation |
|---|---|---|---|---|---|---|---|
| **DLinear** | O(L) | ~12K | Very Low | No | No | ~50% | Already done |
| **NLinear** | O(L) | ~12K | Very Low | No | No | ~50% | Trivial |
| **PatchTST** | O(N^2) | ~200K | Low | No | Via patches | ~50% | Medium |
| **Informer** | O(L log L) | ~500K | Medium | Yes | No | ~50% | High |
| **Autoformer** | O(L log L) | ~400K | Medium | Yes | FFT-based | ~50% | High |
| **iTransformer** | O(F^2) | ~300K | Low-Medium | Yes (attention) | No | ~55% | Medium |
| **TimesNet** | O(L) | ~300K | Medium | Yes | FFT-based | ~50% | High |
| **TSMixer** | O(L*F) | ~150K | Low | Yes (MLP) | No | ~55% | Low-Medium |
| **FEDformer** | O(L) | ~400K | Medium | Yes | FFT-based | ~50% | High |
| **Chronos** | O(L^2) | ~200M | Very High (fine-tune) | Yes | Learned | Unknown | API |
| **S-Mamba** | O(L) | ~200K | Low | Yes | No | ~50% | High |

**Notes:**
- **O(F^2)** for iTransformer: attention across 34 features = 34^2 = 1156 attention weights.
- **O(N^2)** for PatchTST: N = L/patch_size = 6 tokens, so N^2 = 36 (very small).
- **Overfitting risk** assumes 518K samples and 34 features.
- **DirAcc potential** is estimated based on architecture's ability to model feature interactions.

---

## 12. Recommendations for ENERGIVANU

### Tier 1: Implement Now (Low Risk, High Reward)

**1. TSMixer (Recommended First)**
- Why: Simple, captures cross-feature interactions, low overfitting risk.
- Expected improvement: Better DirAcc (55%+), similar or better MAE.
- Implementation: 100-200 lines, standard PyTorch.
- Risk: May not capture periodic patterns.

**2. iTransformer (Recommended Second)**
- Why: Captures feature correlations, proven SOTA, moderate complexity.
- Expected improvement: Better MAE, better DirAcc.
- Implementation: 150-300 lines (mostly transposing input).
- Risk: 34 features may be too few for meaningful attention.

### Tier 2: Implement if Tier 1 Insufficient

**3. PatchTST**
- Why: Patching preserves local patterns, channel independence prevents overfitting.
- Expected improvement: Better long-horizon performance.
- Implementation: 200-400 lines.
- Risk: May not improve DirAcc.

**4. TimesNet**
- Why: Multi-periodicity detection matches our data (8h, 24h cycles).
- Expected improvement: Better periodic pattern capture.
- Implementation: 400-600 lines.
- Risk: Complex, may overfit to synthetic periodicity.

### Tier 3: Advanced / Future Work

**5. S-Mamba (State Space Model)**
- Why: Linear complexity, novel paradigm.
- Risk: Complex implementation, less proven.

**6. Foundation Model Fine-tuning**
- Why: Leverage pre-trained knowledge.
- Risk: Overkill for 518K samples.

### Key Insights for Our Problem

**1. Overfitting Root Cause:**
Our Transformer overfits because:
- Self-attention across 6 time tokens (60/10 patch_size) is too few for meaningful patterns.
- 34 features create many cross-feature parameters.
- Synthetic data is too smooth for complex attention patterns.

**2. Direction Accuracy Root Cause:**
DirAcc is 50% because:
- Current model doesn't explicitly model feature interactions.
- Loss function (MSE) doesn't penalize wrong direction.
- Synthetic data has minimal directional variance.

**3. Best Architecture for Our Case:**
Given 518K samples, 34 features, 60-step lookback:
- **TSMixer** or **iTransformer** for cross-feature learning.
- **DLinear** as baseline (already implemented).
- **PatchTST** if we need better long-horizon performance.

**4. Beyond Architecture:**
Architecture alone won't solve overfitting. Also consider:
- **RevIN** (Reversible Instance Normalization) for non-stationarity.
- **Direction-specific loss** to improve DirAcc.
- **Data augmentation** (jittering, scaling, window slicing).
- **Ensemble** of DLinear + TSMixer + iTransformer.

---

## References

1. Zeng, A., et al. (2023). "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504
2. Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. https://arxiv.org/abs/2211.14730
3. Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021 Best Paper. https://arxiv.org/abs/2012.07436
4. Wu, H., et al. (2021). "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting." NeurIPS 2021. https://arxiv.org/abs/2106.13008
5. Liu, Y., et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024. https://arxiv.org/abs/2310.06625
6. Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023. https://arxiv.org/abs/2210.02186
7. Chen, S.-A., et al. (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting." KDD 2023. https://arxiv.org/abs/2303.06053
8. Zhou, T., et al. (2022). "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting." ICML 2022. https://arxiv.org/abs/2201.12740
9. Kim, T., et al. (2022). "Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift." ICLR 2022.
10. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." https://arxiv.org/abs/2312.00752
11. Ansari, A., et al. (2024). "Chronos: Learning the Language of Time Series." Amazon Science. https://arxiv.org/abs/2403.07815
12. Das, A., et al. (2024). "A Decoder-Only Foundation Model for Time-Series Forecasting." Google Research (TimesFM).

---

## Appendix A: Implementation Priority for ENERGIVANU

```
Week 1: TSMixer implementation + benchmarking
  → 100-200 lines, standard PyTorch
  → Compare with existing DLinear and Transformer

Week 2: iTransformer implementation + benchmarking
  → 150-300 lines, mostly input transposition
  → Test cross-feature attention effectiveness

Week 3: RevIN integration + direction-specific loss
  → 50-100 lines, plug-and-play normalization
  → Custom loss: MSE + lambda * direction_penalty

Week 4: Ensemble (DLinear + TSMixer + iTransformer)
  → Weighted average of predictions
  → Meta-learner for dynamic weighting
```

## Appendix B: Quick Reference - Architecture Diagrams

### DLinear
```
x → [AvgPool] → Trend → [Linear] → Trend_pred
                  ↓
              Seasonal → [Linear] → Seasonal_pred
                                    ↓
                              Sum → output
```

### TSMixer
```
x → [Time-Mix MLP] → [Feature-Mix MLP] → ... → [Linear] → output
     (across L)         (across F)
```

### iTransformer
```
x (B,L,F) → transpose → (B,F,L) → embed → [Attention across F] → [FFN per F] → project → output
```

### PatchTST
```
x (B,L,F) → for each feature f:
              patch → (B,N,P) → embed → [Attention across N] → predict → (B,H)
```
