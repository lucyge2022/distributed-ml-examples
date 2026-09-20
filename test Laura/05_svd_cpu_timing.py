"""CPU SVD timing on Qwen2.5-0.5B-like matrix shapes (synthetic, not loading Qwen)."""
from __future__ import annotations

import time

import numpy as np

print("numpy", np.__version__)

SHAPES = [
    ("q_proj/v_proj", 896, 896),
    ("k_proj (GQA-ish)", 128, 896),
    ("gate/up_proj", 4864, 896),
    ("down_proj", 896, 4864),
    ("toy 2048x2048", 2048, 2048),
    ("toy 4096x4096", 4096, 4096),
]
R = 8


def svd_to_factors(M: np.ndarray, r: int):
    # economy SVD; for tall/wide this is still the dominant cost
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    B = U[:, :r] * np.sqrt(S[:r])
    A = np.sqrt(S[:r])[:, None] * Vh[:r, :]
    return A, B, S


def bench(a: int, b: int, repeats: int = 3) -> float:
    rng = np.random.default_rng(0)
    M = rng.standard_normal((a, b), dtype=np.float64)
    svd_to_factors(M, R)  # warmup
    times = []
    for _ in range(repeats):
        M = rng.standard_normal((a, b), dtype=np.float64)
        t0 = time.perf_counter()
        A, B, _ = svd_to_factors(M, R)
        # touch result so nothing is optimized away
        _ = float((B @ A).sum())
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


print(f"{'name':<20} {'shape':<14} {'avg SVD+factor':>14}")
print("-" * 52)
times = {}
for name, a, b in SHAPES:
    # 4096 is heavy — single timed run after warmup inside bench with repeats=1
    reps = 1 if max(a, b) >= 4096 else 3
    t = bench(a, b, repeats=reps)
    times[(a, b)] = t
    print(f"{name:<20} {str((a, b)):<14} {t:12.4f}s")

qv = times[(896, 896)]
n_layers = 24
print()
print("Rough e2e (NOT what our scripts do today):")
print(f"  SVD q_proj+v_proj on all {n_layers} layers of 0.5B:")
print(f"  ~{n_layers * 2 * qv:.2f}s on this CPU (float64 numpy)")
print("  Complexity: SVD(a×b) ≈ O(min(a*b^2, a^2*b)) — grows fast with width.")
print("  Yes, CPU can do it for 0.5B-sized layers; 7B/70B full-weight SVD is much heavier.")
