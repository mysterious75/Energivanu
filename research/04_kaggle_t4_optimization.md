# Kaggle T4 GPU Optimization for ENERGIVANU

Research document for maximizing training throughput on Kaggle's 2x NVIDIA T4 GPU environment.

---

## 1. NVIDIA T4 GPU Specifications

### Hardware Overview

| Spec | Value |
|------|-------|
| Architecture | Turing (TU104) |
| VRAM | 16 GB GDDR6 |
| CUDA Cores | 2,560 |
| Tensor Cores | 320 (2nd gen) |
| Memory Bandwidth | 320 GB/s |
| L2 Cache | 4 MB |
| TDP | 70 W |
| FP32 Performance | 8.1 TFLOPS |
| FP16 Performance | 65 TFLOPS (with Tensor Cores) |
| INT8 Performance | 130 TOPS (with Tensor Cores) |
| Interconnect (2x T4) | PCIe 3.0 x16 (no NVLink) |

### Key Characteristics for ENERGIVANU

- **FP16 Tensor Core throughput is 8x FP32**: Mixed precision training is not optional -- it is essential. The T4's Tensor Cores deliver 65 TFLOPS in FP16 versus 8.1 TFLOPS in FP32.
- **No NVLink between T4s**: Kaggle provides 2x T4 connected via PCIe, not NVLink. This means inter-GPU communication in DataParallel/DDP is slower than on multi-GPU servers with NVLink. For a ~1M parameter model, the communication overhead is minimal, but it matters for larger models.
- **16 GB VRAM is the bottleneck**: With batch_size=256, lookback=60, horizon=60, num_features=34, each sample is roughly 60x34 = 2,040 floats in FP32 (~8 KB). A batch of 256 samples is ~2 MB input, but intermediate activations during Transformer forward/backward pass consume far more. The 16 GB VRAM ceiling must be respected.
- **70W TDP**: T4 is a low-power GPU. It will not thermal throttle under sustained load in Kaggle's environment, which is a positive.

### VRAM Budget for ENERGIVANU

For the ColossusTransformer (~1M params):
- Model parameters: ~1M x 4 bytes (FP32) = ~4 MB
- With mixed precision copy: ~8 MB total
- Optimizer states (AdamW): 2 x 1M x 4 bytes = ~8 MB
- Gradients: ~4 MB
- **Total model overhead: ~20 MB** (negligible)
- **Activation memory is the real consumer.** With batch_size=256 and Transformer encoder (d_model=128, n_layers=3, n_heads=4, d_ff=512):
  - Self-attention maps: ~batch_size x n_heads x seq_len^2 x n_layers. seq_len = lookback/patch_size = 6 (with patch_size=10). So ~256 x 4 x 36 x 3 = ~110K floats = negligible.
  - FFN intermediates: ~batch_size x seq_len x d_ff x n_layers = 256 x 6 x 512 x 3 = ~2.4M floats = ~10 MB.
  - **Estimated peak VRAM usage: < 1 GB** at batch_size=256. This model is small enough that VRAM is not the constraint -- GPU compute is.

**Recommendation**: The current batch_size=256 is well within T4 VRAM limits. You could increase it to 512 or even 1024 to better saturate the GPU, provided validation loss does not degrade.

---

## 2. Kaggle Platform Constraints

### Session Limits

| Constraint | Value | Impact |
|------------|-------|--------|
| Max session length | 9 hours (GPU) | Hard cutoff; training stops |
| Inactivity timeout | ~10-15 minutes | Session killed if no output/activity |
| GPU count | 2x T4 (when requested) | Use DataParallel or DDP |
| System RAM | ~13 GB | Numpy arrays for 518K samples must fit |
| Disk space | ~20-25 GB (ephemeral) | Data + checkpoints + logs |
| CPU cores | 2-4 cores | Limits DataLoader num_workers |
| Network | Moderate | HF uploads are slow; minimize transfers |
| Internet | Must be enabled in notebook settings | Required for git clone, HF upload |

### Inactivity Timeout Details

Kaggle kills sessions that produce no output for 10-15 minutes. This is the single most common cause of unexpected session termination. The heartbeat mechanism in `kaggle_run.py` (printing every 300 seconds) is correct and necessary.

**Additional safeguards**:
- Print at least one line per epoch (the current `Ep N/M | TL:...` output suffices as long as epochs complete within ~10 minutes).
- If an epoch takes > 10 minutes, add mid-epoch logging or reduce the epoch duration.
- The heartbeat thread runs every 300 seconds (5 minutes), which provides a safety net if an epoch is slow.

### 9-Hour Session Budget

For ENERGIVANU with 120 epochs:
- If each epoch takes ~30 seconds, total = 120 x 30 = 3,600 seconds = 1 hour. Comfortable.
- If each epoch takes ~2 minutes, total = 120 x 2 = 240 minutes = 4 hours. Still within limit.
- If each epoch takes ~4 minutes (unoptimized), total = 120 x 4 = 480 minutes = 8 hours. Tight; risk of timeout.
- **Target: keep epoch time under 2 minutes** to allow for data generation, checkpoint saving, and HF upload overhead.

### System RAM (13 GB) Budget

For 518K samples with lookback=60, horizon=60, num_features=34:
- X: 518K x 60 x 34 x 4 bytes (float32) = ~4.2 GB
- Y: 518K x 60 x 4 bytes = ~124 MB
- S: 518K x 4 bytes = ~2 MB
- D: 518K x 4 bytes = ~2 MB
- **Total: ~4.4 GB**. Fits within 13 GB with room to spare.
- With DataLoader workers loading batches in parallel, add ~200-500 MB per worker.

---

## 3. PyTorch Optimization for T4

### 3.1 Mixed Precision Training (AMP)

This is the single highest-impact optimization for T4. The T4's Tensor Cores provide 8x throughput for FP16 over FP32.

```python
# Add to Trainer.__init__:
self.scaler = torch.cuda.amp.GradScaler()

# Modify Trainer._epoch forward pass:
with torch.cuda.amp.autocast():
    pp, ps, pd = self.model(x)
    l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)

if train:
    self.scaler.scale(l).backward()
    self.scaler.unscale_(self.opt)
    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
    self.scaler.step(self.opt)
    self.scaler.update()
    self.opt.zero_grad()
```

**Expected speedup**: 1.5-3x depending on the operation mix. For the Transformer model with its self-attention and FFN layers, expect ~2x speedup. For DLinear (mostly linear layers), expect ~1.5x.

**Caveats**:
- The SpikeLoss uses `tp.std()` which can be numerically unstable in FP16. The `autocast` context handles this by keeping reduction operations in FP32, but verify no NaN losses appear.
- Gradient clipping must happen after `unscale_()` and before `step()`.
- The `GradScaler` is essential: without it, FP16 gradients underflow for small learning rates like 5e-6.

### 3.2 torch.compile() (PyTorch 2.0+)

Kaggle typically provides PyTorch 2.0+. `torch.compile` fuses operations and reduces Python overhead.

```python
# After model creation, before wrapping in DataParallel:
model = torch.compile(model, mode="reduce-overhead")
```

**Modes**:
- `"default"`: Good balance of compile time and runtime speedup.
- `"reduce-overhead"`: Best for small models with many small operations. Recommended for ENERGIVANU.
- `"max-autotune"`: Longer compile time, best runtime. Only worthwhile if you have time budget.

**Expected speedup**: 10-30% for small models. The compile step takes 1-3 minutes on first call, which is a one-time cost.

**Important**: `torch.compile` must be applied BEFORE `nn.DataParallel` wrapping. The current code in `kaggle_run.py` wraps in DataParallel inside `Trainer.__init__`, so you would need to either:
1. Compile the model before passing it to Trainer, or
2. Modify Trainer to accept a `compile=True` flag.

### 3.3 Optimal Batch Size

Current batch_size=256. T4 has 16 GB VRAM and the model is ~1M params.

| Batch Size | VRAM Usage (est.) | GPU Utilization | Epoch Time |
|------------|-------------------|-----------------|------------|
| 64 | ~200 MB | Low (~30%) | Slow |
| 128 | ~350 MB | Medium (~50%) | Moderate |
| 256 | ~600 MB | Good (~70%) | Good |
| 512 | ~1.1 GB | Better (~80%) | Better |
| 1024 | ~2.0 GB | High (~85%) | Best |
| 2048 | ~3.8 GB | High (~85%) | Similar |

**Recommendation**: Increase batch_size to 512 or 1024. The model's activation memory is small, and larger batches better saturate the T4's Tensor Cores. However, larger batches may require learning rate scaling (see Section 4.4).

### 3.4 DataLoader Optimization

Current settings in `trainer.py`:
```python
DataLoader(tds, tc.batch_size, shuffle=True, num_workers=self.num_workers,
           pin_memory=True, prefetch_factor=2)
```

**Recommended settings for Kaggle T4**:
```python
DataLoader(tds, tc.batch_size, shuffle=True,
           num_workers=2,           # Kaggle has 2-4 CPU cores; 2 is optimal
           pin_memory=True,         # Fast CPU-to-GPU transfer
           prefetch_factor=4,       # Increase from 2 to 4 for more prefetching
           persistent_workers=True) # Keep workers alive between epochs
```

**Key points**:
- `num_workers=2`: Kaggle has limited CPU cores. Setting num_workers too high (e.g., 4-8) causes context switching overhead and can OOM on system RAM. The current `num_workers=2` in `kaggle_run.py` is correct.
- `persistent_workers=True`: This avoids the cost of spawning new worker processes each epoch. The current code does NOT use this -- adding it can save 2-5 seconds per epoch.
- `prefetch_factor=4`: Default is 2. Increasing to 4 means each worker prefetches 4 batches ahead, reducing GPU idle time.
- `pin_memory=True`: Already set. This uses pinned (page-locked) memory for faster CPU-to-GPU transfers via DMA.

### 3.5 Gradient Accumulation vs. Larger Batch

If you want an effective batch size of 1024 but VRAM only allows 256 per step:
```python
accumulation_steps = 4  # 256 x 4 = 1024 effective batch

for i, (x, yp, ys, yd) in enumerate(dl):
    with torch.cuda.amp.autocast():
        pp, ps, pd = model(x)
        l, m = loss_fn(pp, yp, ps, ys, pd, yd)
        l = l / accumulation_steps

    scaler.scale(l).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
```

**For ENERGIVANU**: Since the model is small (~1M params, < 1 GB VRAM at batch_size=1024), gradient accumulation is unnecessary. Just increase the batch size directly. Gradient accumulation adds complexity and slightly slows training due to more forward/backward passes.

### 3.6 DataParallel vs. DistributedDataParallel

| Feature | DataParallel (DP) | DistributedDataParallel (DDP) |
|---------|-------------------|-------------------------------|
| Setup complexity | Low | Medium |
| GIL contention | Yes (single process) | No (multi-process) |
| GPU utilization | ~70-80% | ~90-95% |
| Load balancing | Uneven (GPU 0 does more) | Even |
| Recommended for | Quick testing | Production training |

**For ENERGIVANU with 2x T4**:
- Current code uses `nn.DataParallel`. This is fine for a ~1M param model where communication overhead is minimal.
- DDP would give ~10-15% better throughput, but requires launching with `torchrun` or `torch.multiprocessing.spawn`, which complicates Kaggle notebook execution.
- **Recommendation**: Stick with DataParallel for simplicity. The model is small enough that the GIL bottleneck is negligible. If you hit time limits, switch to DDP.

**DDP setup for Kaggle** (if needed):
```python
import torch.distributed as dist
import torch.multiprocessing as mp

def train_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = ColossusTransformer(cfg.model).cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # ... training loop ...
    dist.destroy_process_group()

mp.spawn(train_ddp, args=(2,), nprocs=2)
```

### 3.7 Channels-Last Memory Format

For convolutional models, `model = model.to(memory_format=torch.channels_last)` can give 10-20% speedup. For the ENERGIVANU Transformer and DLinear models, which use Linear layers and attention (not convolutions), channels-last is not applicable and provides no benefit.

---

## 4. Training Optimization Techniques

### 4.1 Gradient Checkpointing

Gradient checkpointing trades compute for memory by recomputing activations during backward pass instead of storing them. For a ~1M param model using < 1 GB VRAM, this is **not needed** and would only slow training.

If the model were larger (e.g., 50M+ params), you would add:
```python
from torch.utils.checkpoint import checkpoint

class TransformerEncoder(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**For ENERGIVANU**: Skip this. The model is too small to benefit.

### 4.2 Efficient Attention

The current model uses `nn.TransformerEncoderLayer` with standard multi-head attention. For seq_len=6 (lookback=60 / patch_size=10), the attention matrix is only 6x6, which is tiny.

**FlashAttention**: Provides speedup for long sequences (256+ tokens). With seq_len=6, FlashAttention overhead actually makes it slower than standard attention. Do NOT use FlashAttention for this model.

**If you increase patch_size** (e.g., patch_size=5, seq_len=12): Still too short for FlashAttention to matter.

**Recommendation**: Standard attention is optimal for this sequence length.

### 4.3 torch.compile Detailed Settings

```python
# Option 1: Compile full model (simplest)
model = torch.compile(model, mode="reduce-overhead")

# Option 2: Compile with specific backend
model = torch.compile(model, backend="inductor", mode="reduce-overhead")

# Option 3: Selective compilation (compile only expensive layers)
# Not recommended for a 1M param model -- compile everything.
```

**The `inductor` backend** (default in PyTorch 2.0+) generates optimized Triton kernels for GPU. It can fuse the attention computation, FFN, and activation functions into fewer kernel launches, reducing the overhead of launching many small CUDA kernels -- a significant bottleneck for small models.

### 4.4 Learning Rate Scaling

Current config: lr=5e-6, batch_size=256.

When increasing batch size, learning rate should be scaled:

| Effective Batch Size | Linear Scaling LR | Sqrt Scaling LR |
|---------------------|-------------------|-----------------|
| 256 (current) | 5e-6 | 5e-6 |
| 512 | 1e-5 | 7.07e-6 |
| 1024 | 2e-5 | 1e-5 |
| 2048 | 4e-5 | 1.41e-5 |

**Linear scaling rule**: `new_lr = base_lr * (new_batch_size / base_batch_size)`
**Sqrt scaling rule**: `new_lr = base_lr * sqrt(new_batch_size / base_batch_size)`

**Recommendation for ENERGIVANU**: Use sqrt scaling. Linear scaling is aggressive and can cause training instability, especially with the asymmetric SpikeLoss. If moving from batch_size=256 to 512, try lr=7e-6.

### 4.5 Optimal Learning Rate for Large Batch

The current lr=5e-6 is quite conservative. With warmup=500 steps and cosine decay, this is a safe choice. When scaling up batch size:

1. Start with sqrt-scaled LR.
2. Monitor loss curve for the first 5-10 epochs.
3. If loss oscillates, reduce LR by 30%.
4. If loss plateaus early, increase LR by 30%.

**LAMB optimizer** is designed for large-batch training but adds complexity. For a 1M param model, AdamW with proper LR scaling is sufficient.

---

## 5. Kaggle-Specific Tricks

### 5.1 Heartbeat Mechanism

The current implementation in `kaggle_run.py` is correct:
```python
def _heartbeat():
    while True:
        time.sleep(300)  # Every 5 minutes
        print(f"  [heartbeat] {time.strftime('%H:%M:%S')} — training running", flush=True)
_th.Thread(target=_heartbeat, daemon=True).start()
```

**Improvements**:
1. **Flush output**: Already uses `flush=True`. Good.
2. **Include metrics in heartbeat**: Print current epoch, loss, and estimated time remaining. This helps debug if the session does timeout.
3. **Consider reducing interval to 120 seconds**: 5 minutes is fine for normal operation, but if an epoch takes 8+ minutes and there is no other output, the gap between the last epoch print and the next could approach 10 minutes.

```python
def _heartbeat():
    while True:
        time.sleep(120)  # Every 2 minutes
        elapsed = time.time() - t_start
        print(f"  [heartbeat] {time.strftime('%H:%M:%S')} | "
              f"elapsed: {elapsed/3600:.1f}h | training active", flush=True)
_th.Thread(target=_heartbeat, daemon=True).start()
```

### 5.2 Checkpoint Saving Strategy

Current strategy: Save `best.pt` whenever validation loss improves, plus periodic checkpoints every `save_every` epochs.

**Recommended improvements**:

```python
# 1. Save checkpoint to Kaggle's /kaggle/working (persistent within session)
#    AND to /kaggle/temp (faster writes, but lost on disconnect)
CKPT_DIR = "/kaggle/working/checkpoints"  # Persistent within session
FAST_DIR = "/kaggle/temp/checkpoints"      # Fast local storage

# 2. Save every epoch (not just every N epochs)
#    For a 1M param model, checkpoint is ~4 MB. Saving every epoch is cheap.

# 3. Save training state comprehensively:
checkpoint = {
    "ep": ep,
    "model": model.state_dict(),
    "opt": opt.state_dict(),
    "scaler": scaler.state_dict(),  # If using AMP
    "y_mean": y_mean,
    "y_std": y_std,
    "history": history,
    "best_val_loss": best,
    "rng_state": torch.random.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all(),
}
torch.save(checkpoint, f"{CKPT_DIR}/checkpoint_ep{ep}.pt")
```

### 5.3 Resume from Checkpoint After Disconnection

The current resume logic in `kaggle_run.py` loads from `best.pt`. This is correct but loses the optimizer state momentum. To properly resume:

```python
# Load full training state
ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["opt"])
if "scaler" in ckpt:
    scaler.load_state_dict(ckpt["scaler"])
resume_ep = ckpt["ep"]
```

**The current code already does this correctly** (lines 165-168 of `kaggle_run.py`). The only missing piece is saving/loading the scaler state if AMP is added.

### 5.4 Memory Management

```python
import gc

# After data loading is complete and arrays are no longer needed in CPU memory:
# (Keep only what DataLoader needs)

# Clear CUDA cache between training runs or after large operations:
torch.cuda.empty_cache()

# Force garbage collection periodically:
gc.collect()

# Monitor memory usage:
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
```

**For ENERGIVANU specifically**:
- After data generation and feature engineering, the raw dataframe (`df`) can be deleted to free ~1-2 GB of system RAM.
- The numpy arrays X, Y, S, D must stay in memory for DataLoader access. At ~4.4 GB total, this is manageable.
- After creating the TensorDataset, the numpy arrays could theoretically be freed if the DataLoader has loaded them into tensors, but PyTorch shares memory with numpy arrays, so this is not straightforward.

### 5.5 Disk Space Management

```python
# Kaggle provides ~20-25 GB ephemeral disk
# Check available space:
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"Disk: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")

# Keep only the last N checkpoints to save disk space:
import glob
ckpts = sorted(glob.glob(f"{CKPT_DIR}/checkpoint_ep*.pt"))
if len(ckpts) > 5:
    for old in ckpts[:-5]:
        os.remove(old)
```

**Disk budget**:
- Git repo: ~50 MB
- Generated data (parquet + npy): ~1 GB
- Checkpoints (120 epochs x 4 MB): ~500 MB (if saving all; keep only last 5-10)
- Logs/outputs: ~100 MB
- **Total: ~2 GB**. Well within limits.

### 5.6 Handling Session Disconnections

Kaggle sessions can disconnect due to:
1. Inactivity timeout (prevent with heartbeat)
2. 9-hour limit (unavoidable; save checkpoints frequently)
3. Platform issues (rare; save checkpoints frequently)

**Best practice**: Design the training loop to be fully resumable. The current implementation does this. On re-run:
1. Script detects existing checkpoint at `{CKPT_DIR}/best.pt`.
2. Loads model weights and optimizer state.
3. Resumes from the saved epoch.

**One improvement**: Save the latest checkpoint (not just best) to enable resuming even if the best checkpoint is from a much earlier epoch:
```python
# Save latest checkpoint every epoch:
torch.save(checkpoint, f"{CKPT_DIR}/latest.pt")

# On resume, prefer latest over best for training continuity:
RESUME_CKPT = f"{CKPT_DIR}/latest.pt" if os.path.exists(f"{CKPT_DIR}/latest.pt") else f"{CKPT_DIR}/best.pt"
```

---

## 6. Common Pitfalls

### 6.1 OOM Errors

**VRAM OOM**:
- Cause: Batch size too large, or model too large.
- For ENERGIVANU (~1M params): Unlikely with batch_size <= 2048.
- Fix: Reduce batch size, enable gradient checkpointing (for larger models), or use gradient accumulation.

**System RAM OOM**:
- Cause: Too many DataLoader workers, or data arrays too large.
- For ENERGIVANU (~4.4 GB data): Should be fine with 13 GB system RAM.
- Fix: Reduce num_workers from 2 to 1 or 0. Use memory-mapped numpy arrays (`np.load(..., mmap_mode='r')`).

**Detection and monitoring**:
```python
# Check VRAM usage:
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB / 16 GB")

# Check system RAM:
import psutil
ram = psutil.virtual_memory()
print(f"RAM: {ram.used/1e9:.1f} GB / {ram.total/1e9:.1f} GB ({ram.percent}%)")
```

### 6.2 DataLoader num_workers Issues

**Linux vs. Windows vs. Kaggle**:
- Kaggle runs Linux. `num_workers > 0` works correctly with fork-based multiprocessing.
- On Windows, `num_workers > 0` requires `if __name__ == '__main__':` guard.
- `num_workers=0` means data loading happens in the main process, which can bottleneck GPU utilization.
- `num_workers=2` is a good default for Kaggle's 2-4 CPU cores.
- `persistent_workers=True` prevents worker re-spawning each epoch.

**If you see "Too many open files" error**:
```python
# Reduce num_workers or increase system limit:
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
```

### 6.3 DataParallel Overhead

For a ~1M param model on 2x T4:
- DataParallel splits each batch across GPUs and gathers results on GPU 0.
- The gather operation involves PCIe transfer (~16 GB/s bidirectional).
- For a 256-sample batch, the gather transfers ~600 MB of activations. At 16 GB/s, this takes ~37 ms.
- If each batch takes 200 ms total, the DP overhead is ~18%. Acceptable.
- **If the model were smaller** (e.g., 100K params), the DP overhead could exceed the compute time, making single-GPU training faster.

**For ENERGIVANU**: DataParallel is fine. The model is large enough that compute dominates over communication.

**When NOT to use DataParallel**:
- Model < 500K params
- Batch size < 32
- When training time is dominated by data loading (not GPU compute)

### 6.4 Gradient Accumulation Pitfalls

1. **Forgetting to zero gradients**: With accumulation, `zero_grad()` must happen AFTER the optimizer step, not before each forward pass. The current ENERGIVANU code calls `zero_grad()` before each forward pass, which is correct for non-accumulation training.

2. **Loss scaling**: When accumulating, divide loss by accumulation_steps before backward:
   ```python
   loss = loss / accumulation_steps
   ```
   Forgetting this causes gradient magnitudes to be accumulation_steps times too large.

3. **Logging frequency**: Metrics are computed per micro-batch, not per effective batch. If logging every step, metrics will be noisy.

4. **BatchNorm statistics**: With accumulation, BatchNorm computes statistics per micro-batch, not effective batch. This can cause instability. Use GroupNorm or LayerNorm instead (the ENERGIVANU model already uses LayerNorm, so this is fine).

### 6.5 Mixed Precision Gotchas

1. **Loss scaling too aggressive**: If you see NaN losses, the GradScaler may be choosing too high a scale factor. Monitor with:
   ```python
   print(f"  Scale: {scaler.get_scale()}")
   ```

2. **Operations that must stay in FP32**:
   - Loss computation (especially cross-entropy with class weights)
   - Gradient clipping
   - Softmax (handled by autocast, but verify)
   - Layer normalization (handled by autocast)

3. **Model outputs in FP16**: The SpikeLoss computes `tp.std()` which can underflow in FP16 for small values. The autocast context should handle this, but if you see NaN, wrap loss computation in `with torch.cuda.amp.autocast(enabled=False):` and manually cast inputs to FP32.

### 6.6 Kaggle-Specific Pitfalls

1. **Internet must be enabled**: In notebook settings, "Internet" must be ON for git clone and HF upload. Without this, the script will hang at `git clone`.

2. **GPU persistence mode**: Kaggle does not support `nvidia-smi -pm 1`. GPU state resets between sessions.

3. **Output directory**: `/kaggle/working/` is the writable directory. `/kaggle/input/` is read-only. Do not try to write to `/kaggle/input/`.

4. **Session restart clears /kaggle/working/**: All files in `/kaggle/working/` are lost when the session ends. Upload important checkpoints to HuggingFace or Kaggle Datasets before the session ends.

5. **Concurrent sessions**: Kaggle allows only one GPU session at a time. If you start a new session, the old one is killed.

---

## 7. Specific Recommendations for ENERGIVANU

### Current Configuration Analysis

| Setting | Current | Recommended | Rationale |
|---------|---------|-------------|-----------|
| batch_size | 256 | 512 | Better GPU saturation; model uses < 1 GB VRAM |
| lr | 5e-6 | 7e-6 | Sqrt scaling for batch_size 512 |
| num_workers | 2 | 2 (keep) | Kaggle has limited CPU cores |
| pin_memory | True | True (keep) | Already optimal |
| prefetch_factor | 2 | 4 | More prefetching reduces GPU idle time |
| persistent_workers | not set | True | Saves 2-5s per epoch |
| Mixed Precision | Not used | Enable AMP | 1.5-2x speedup on T4 Tensor Cores |
| torch.compile | Not used | Enable | 10-30% speedup |
| DataParallel | Used | Keep | Fine for 1M param model |
| Gradient accumulation | Not used | Not needed | Batch size fits in VRAM |

### Implementation Priority

**Phase 1: High Impact, Low Effort**
1. Enable AMP (mixed precision) -- ~2x speedup
2. Add `persistent_workers=True` to DataLoaders -- minor but free
3. Increase `prefetch_factor` to 4 -- minor but free

**Phase 2: Medium Impact, Medium Effort**
4. Add `torch.compile(model, mode="reduce-overhead")` -- ~10-30% speedup
5. Increase batch_size to 512 with sqrt LR scaling -- better GPU utilization
6. Save latest checkpoint in addition to best -- better resume after disconnection

**Phase 3: Low Impact or High Effort**
7. Switch from DataParallel to DDP -- ~10-15% speedup, but complicates code
8. Profile with `torch.profiler` to identify remaining bottlenecks
9. Consider reducing patch_size from 10 to 5 for longer sequences (changes model behavior)

### Expected Training Time

With current config (no AMP, batch_size=256):
- ~518K samples / 256 = ~2,023 batches per epoch
- Assume ~0.5 ms per batch = ~1 second per epoch (optimistic)
- 120 epochs = ~2 minutes
- With overhead: ~5-10 minutes total

With optimized config (AMP, batch_size=512, torch.compile):
- ~518K samples / 512 = ~1,012 batches per epoch
- With AMP ~0.3 ms per batch = ~0.3 seconds per epoch
- 120 epochs = ~1 minute
- With overhead: ~3-5 minutes total

The model is small enough that training is not the bottleneck. Data generation and feature engineering may take longer than actual training.

### Complete Optimized Training Loop

```python
# ─── After model creation ───────────────────────────────
# Compile BEFORE DataParallel
model = torch.compile(model, mode="reduce-overhead")

# ─── In Trainer.__init__ ───────────────────────────────
self.scaler = torch.cuda.amp.GradScaler()

# ─── In Trainer._epoch ─────────────────────────────────
def _epoch(self, dl, train, total=0):
    self.model.train() if train else self.model.eval()
    s = {k:0. for k in ["pl","sl","dl","loss","mae","da","sa"]}; n=0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, yp, ys, yd in dl:
            x = x.to(self.dev, non_blocking=True)
            yp = yp.to(self.dev, non_blocking=True)
            ys = ys.to(self.dev, non_blocking=True)
            yd = yd.to(self.dev, non_blocking=True)

            if train:
                self.step += 1
                for g in self.opt.param_groups:
                    g["lr"] = self._lr(self.step, total)

            with torch.cuda.amp.autocast():
                pp, ps, pd = self.model(x)
                l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)

            if train:
                self.opt.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                self.scaler.scale(l).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()

            for k in s: s[k] += m[k]
            n += 1
    return {k: v/max(n,1) for k, v in s.items()}

# ─── DataLoader creation ───────────────────────────────
tdl = DataLoader(tds, tc.batch_size, shuffle=True,
                 num_workers=2, pin_memory=True,
                 prefetch_factor=4, persistent_workers=True)

# ─── Checkpoint saving ─────────────────────────────────
checkpoint = {
    "ep": ep,
    "model": model_raw.state_dict() if is_parallel else model.state_dict(),
    "opt": opt.state_dict(),
    "scaler": scaler.state_dict(),
    "y_mean": y_mean, "y_std": y_std,
    "history": hist,
}
torch.save(checkpoint, f"{CKPT_DIR}/latest.pt")
if vm["loss"] < best:
    torch.save(checkpoint, f"{CKPT_DIR}/best.pt")
```

---

## 8. Monitoring and Debugging

### GPU Utilization Monitoring

```python
# Run in a separate thread during training:
import subprocess

def monitor_gpu():
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        for i, line in enumerate(result.stdout.strip().split("\n")):
            gpu_util, mem_util, mem_used, mem_total, temp = line.split(", ")
            print(f"  GPU{i}: {gpu_util}% util, {mem_used}/{mem_total}MB mem, {temp}C")
        time.sleep(30)
```

### Training Speed Metrics

```python
# Add to training loop:
samples_per_sec = len(dl.dataset) / epoch_time
batches_per_sec = len(dl) / epoch_time
gpu_util_estimate = (batch_time / (batch_time + data_loading_time)) * 100

print(f"  Speed: {samples_per_sec:.0f} samples/s, {batches_per_sec:.1f} batches/s")
```

### Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch too large | Reduce batch_size |
| `RuntimeError: NCCL` | Multi-GPU communication failure | Check GPU visibility, try DDP backend="gloo" |
| `UserWarning: grad` | Gradient not contiguous | Add `.contiguous()` before backward |
| `NaN in loss` | FP16 underflow or bad data | Check data for NaN, reduce LR, check scaler |
| `Slow training` | CPU bottleneck | Increase num_workers, use persistent_workers |
| `Session timeout` | No output for 10+ min | Reduce heartbeat interval, add mid-epoch logging |

---

## 9. Summary of Actionable Changes

### To `kaggle_run.py`:

1. Add `import torch.cuda.amp` and GradScaler setup.
2. Compile model before Trainer: `model = torch.compile(model, mode="reduce-overhead")`.
3. Increase `BATCH_SIZE` from 256 to 512 and `LR` from 5e-6 to 7e-6.
4. Reduce heartbeat interval from 300s to 120s.

### To `src/engine/trainer.py`:

1. Add `self.scaler = torch.cuda.amp.GradScaler()` in `__init__`.
2. Wrap forward pass in `torch.cuda.amp.autocast()`.
3. Use `scaler.scale(l).backward()`, `scaler.unscale_()`, `scaler.step()`, `scaler.update()`.
4. Add `persistent_workers=True` and `prefetch_factor=4` to DataLoaders.
5. Use `opt.zero_grad(set_to_none=True)` instead of `opt.zero_grad()`.
6. Add `non_blocking=True` to `.to(device)` calls for overlapped transfer.
7. Save `scaler.state_dict()` in checkpoints.
8. Save `latest.pt` in addition to `best.pt`.

### Expected Combined Speedup

| Optimization | Speedup Factor |
|--------------|---------------|
| AMP (FP16) | 1.5-2.0x |
| torch.compile | 1.1-1.3x |
| Larger batch (256->512) | 1.1-1.2x |
| persistent_workers | 1.05x |
| non_blocking transfers | 1.02x |
| **Combined** | **~2.0-3.0x** |

With these optimizations, a 120-epoch training run should complete in 3-5 minutes on Kaggle's 2x T4, well within the 9-hour session limit. The primary remaining bottleneck will be data generation and feature engineering, not GPU training.
