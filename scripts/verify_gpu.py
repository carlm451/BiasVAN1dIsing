"""
Quick GPU sanity check: train a small VAN on cuda:0 to verify
the full pipeline works before launching a multi-GPU sweep.

Usage:
    python scripts/verify_gpu.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.van.train import train, TrainConfig
from src.exact.analytical_formulas import free_energy_per_spin


def verify(N, K, h, batch_size=4000, max_step=500, device='cuda'):
    """Train one VAN and compare to exact result."""
    print(f"\n--- N={N}, K={K}, h={h}, device={device}, batch_size={batch_size} ---")

    # Exact reference
    exact_f = free_energy_per_spin(K, h, N)
    print(f"  Exact beta*f/N = {exact_f:.8f}")

    # Train VAN
    tc = TrainConfig(
        N=N, K=K, h=h,
        use_bias=True,
        z2=(abs(h) < 1e-10),
        batch_size=batch_size,
        lr=0.01,
        max_step=max_step,
        seed=42,
        device=device,
    )

    t0 = time.time()
    result = train(tc)
    elapsed = time.time() - t0

    error = result.final_free_energy - exact_f
    print(f"  VAN   beta*f/N = {result.final_free_energy:.8f}")
    print(f"  Error          = {error:+.2e}")
    print(f"  Converged: {result.converged} at step {result.final_step}")
    print(f"  Time: {elapsed:.1f}s")

    return abs(error) < 0.01  # Loose threshold for quick check


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test on GPU 0
    results = []
    results.append(verify(N=8,  K=1.0, h=0.5, batch_size=4000, max_step=500))
    results.append(verify(N=32, K=0.5, h=0.2, batch_size=4000, max_step=1000))

    print("\n" + "=" * 50)
    if all(results):
        print("All checks PASSED — GPU pipeline is working")
    else:
        print("Some checks FAILED — investigate before running full sweep")
        sys.exit(1)


if __name__ == "__main__":
    main()
