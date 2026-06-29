"""
Energivanu - Claims Verification Script
=======================================
Verifies benchmarks and metrics claimed in the README.
"""

import os
import sys
import numpy as np
import torch
import time

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from energivanu import EnergivanuPEB, MPCController, PeakShavingOptimizer, PhaseStaggeringScheduler

def print_section(title):
    print("=" * 70)
    print(f" VERIFYING: {title}")
    print("=" * 70)

def verify_model_parameters():
    print_section("Model Parameters & Size")
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    params = model.count_parameters()
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Configured Features: {model.n_features}")
    print(f"Sequence Length: {model.seq_len}")
    print(f"Prediction Horizon: {model.pred_horizon}")
    print(f"Calculated parameters: {params:,}")
    assert params == 338252, f"Parameter mismatch: {params}"
    print("[OK] SUCCESS: Model parameters match exactly 338,252.")
    print()

def verify_mpc_smoothing():
    print_section("BESS Battery Smoothing (Sinusoidal Load Trace)")
    mpc = MPCController()
    # 30-step sinusoidal trace matching proof
    trace = np.sin(np.linspace(0, 4 * np.pi, 30)) * 50 + 200
    result = mpc.simulate(trace)
    smoothing = result["metrics"]["smoothing_percentage"]
    print(f"Input load trace points: {len(trace)}")
    print(f"Target power level: {result['metrics']['target_power_mw']} MW")
    print(f"Calculated smoothing: {smoothing:.2f}%")
    # Verify smoothing is 30.0% (allowing small float delta)
    assert abs(smoothing - 30.0) < 0.1, f"Expected 30.0%, got {smoothing}"
    print("[OK] SUCCESS: Battery smoothing metric matches 30.0%.")
    print()

def verify_peak_shaving():
    print_section("Time-of-Use Peak Shaving Optimization")
    opt = PeakShavingOptimizer()
    hourly = list(np.sin(np.linspace(0, 2 * np.pi, 24)) * 100 + 200)
    result = opt.simulate_month(hourly)
    reduction = result["peak_reduction_pct"]
    print(f"Baseline peak: {result['peak_before_mw']} MW")
    print(f"Shaved peak: {result['peak_after_mw']} MW")
    print(f"Calculated peak reduction: {reduction:.2f}%")
    print(f"Estimated Monthly Savings: ${result['total_monthly_demand_savings_usd']:,.2f}")
    assert abs(reduction - 10.5) < 0.1, f"Expected 10.5%, got {reduction}"
    print("[OK] SUCCESS: Peak demand reduction matches 10.5%.")
    print()

def verify_phase_staggering():
    print_section("GPU Cluster Phase Staggering Scheduler")
    sched = PhaseStaggeringScheduler()
    # Fix seed for reproducibility of random noise in schedule
    np.random.seed(42)
    result = sched.schedule_clusters(n_clusters=4, n_steps=8640)
    reduction = result["std_reduction_pct"]
    print(f"Unstaggered aggregate variance (Std): {result['baseline_std_mw']} MW")
    print(f"Staggered aggregate variance (Std): {result['stagger_std_mw']} MW")
    print(f"Aggregate volatility reduction: {reduction:.2f}%")
    assert abs(reduction - 59.0) < 1.0, f"Expected ~59.0%, got {reduction}"
    print("[OK] SUCCESS: Volatility standard deviation reduction matches ~59.0%.")
    print()

def verify_onnx_speed():
    print_section("ONNX Runtime Serialization & Speedup")
    onnx_path = os.path.join(os.path.dirname(__file__), "energivanu.onnx")
    if not os.path.exists(onnx_path):
        print(f"[SKIP] ONNX file not found at {onnx_path}. Run ONNX export script to generate.")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime not installed in this python environment.")
        return

    # Load ONNX and PyTorch model
    sess = ort.InferenceSession(onnx_path)
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    model.eval()

    dummy = np.random.randn(1, 30, 15).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy)

    # Warmup
    for _ in range(10):
        model(dummy_torch)
        sess.run(None, {"power_history": dummy})

    # Benchmark PyTorch
    t0 = time.time()
    for _ in range(100):
        model(dummy_torch)
    pt_ms = (time.time() - t0) * 10

    # Benchmark ONNX
    t0 = time.time()
    for _ in range(100):
        sess.run(None, {"power_history": dummy})
    onnx_ms = (time.time() - t0) * 10

    speedup = pt_ms / onnx_ms
    print(f"PyTorch CPU Latency: {pt_ms:.2f} ms")
    print(f"ONNX CPU Latency: {onnx_ms:.2f} ms")
    print(f"Calculated Speedup: {speedup:.1f}x")
    print("[OK] SUCCESS: ONNX serialization speed benchmark complete.")
    print()

def main():
    print("======================================================================")
    print("                  ENERGIVANU BENCHMARK VERIFICATION REPORT            ")
    print("======================================================================")
    print()
    
    try:
        verify_model_parameters()
        verify_mpc_smoothing()
        verify_peak_shaving()
        verify_phase_staggering()
        verify_onnx_speed()
        
        print("======================================================================")
        # Verify if York checkpoint is available locally to inform user
        ckpt_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "src", "models", "checkpoints", "best_model_real.pt"
        ))
        if os.path.exists(ckpt_path):
            print("[STATUS] 'best_model_real.pt' found. Model trained on York University data.")
        else:
            print("[INFO] 'best_model_real.pt' not found (gitignored). Running on synthetic weights.")
            
        print("[SUCCESS] ALL VERIFIABLE CLAIMS CONFLICT-FREE & VALIDATED SUCCESSFULLY!")
        print("======================================================================")
    except AssertionError as e:
        print(f"[FAIL] CLAIMS VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
