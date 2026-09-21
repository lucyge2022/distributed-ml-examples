"""
LoRA as low-rank factorization (the "eigen-ish" idea, without SVD).

For a frozen weight matrix W of shape (a, b), a full fine-tune would learn
an update ΔW also of shape (a, b) — that's a*b trainable numbers.

LoRA instead parameterizes the update as a product of two skinny matrices:

    ΔW ≈ B @ A
         ^     ^
         |     +-- A has shape (r, b)   # "down-projection"
         +-------- B has shape (a, r)   # "up-projection"

So you only train a*r + r*b parameters. When r << min(a, b), that is a huge
saving. At inference:

    y = x @ (W + ΔW).T     # or whatever your layout is
      = x @ W.T + x @ A.T @ B.T

This is *related* to the idea of keeping only the top-r singular directions
of ΔW (SVD / "eigenvectors of ΔWᵀΔW"), but during training we do not run SVD —
we just learn A and B with gradient descent, which finds some useful rank-r
subspace on its own.

Run:
    python 01_lora_math.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_params(t: torch.Tensor) -> int:
    return t.numel()


def demo_shapes(a: int = 4096, b: int = 4096, r: int = 8) -> None:
    print("=" * 60)
    print("1) Shape / parameter count")
    print("=" * 60)

    W = torch.randn(a, b)
    A = torch.randn(r, b) / (r**0.5)  # (r, b)
    B = torch.zeros(a, r)  # (a, r) — zero init so ΔW starts at 0

    delta_full = torch.randn(a, b)
    delta_lora = B @ A  # (a, r) @ (r, b) -> (a, b)

    print(f"W shape:           {tuple(W.shape)}   params={count_params(W):,}")
    print(f"full ΔW shape:     {tuple(delta_full.shape)}   params={count_params(delta_full):,}")
    print(f"LoRA A shape:      {tuple(A.shape)}      params={count_params(A):,}")
    print(f"LoRA B shape:      {tuple(B.shape)}      params={count_params(B):,}")
    print(f"LoRA ΔW = B@A:     {tuple(delta_lora.shape)}   trainable={count_params(A)+count_params(B):,}")
    print(
        f"compression vs full ΔW: "
        f"{(count_params(A)+count_params(B)) / count_params(delta_full):.4%}"
    )
    print()


def demo_svd_intuition(a: int = 64, b: int = 64, r: int = 4) -> None:
    print("=" * 60)
    print("2) SVD intuition: best rank-r approximation of a ΔW")
    print("=" * 60)

    torch.manual_seed(0)
    # Pretend some full fine-tune produced this dense update:
    delta = torch.randn(a, b)
    # Force it to be approximately low-rank so the story is clear:
    true_A = torch.randn(r, b)
    true_B = torch.randn(a, r)
    delta = true_B @ true_A + 0.01 * torch.randn(a, b)

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    # Best rank-r approx via truncated SVD (the "eigen" picture):
    delta_r = (U[:, :r] * S[:r]) @ Vh[:r, :]

    # Factor it exactly like LoRA: B=(a,r), A=(r,b)
    B_svd = U[:, :r] * S[:r].sqrt()
    A_svd = S[:r].sqrt().unsqueeze(1) * Vh[:r, :]
    assert torch.allclose(B_svd @ A_svd, delta_r, atol=1e-5)

    err_full_vs_true = (delta - true_B @ true_A).norm() / delta.norm()
    err_svd = (delta - delta_r).norm() / delta.norm()
    print(f"relative noise in synthetic ΔW: {err_full_vs_true:.4f}")
    print(f"relative error of rank-{r} SVD approx: {err_svd:.4f}")
    print(f"reconstructed with B@A shapes {tuple(B_svd.shape)} @ {tuple(A_svd.shape)}")
    print()


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper around nn.Linear — same math PEFT uses."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        # Freeze the big matrix
        for p in self.base.parameters():
            p.requires_grad = False

        a, b = base.out_features, base.in_features  # W is (a, b)
        # A: (r, b), B: (a, r)  — matches "a×b ≈ (a×r)(r×b)"
        self.lora_A = nn.Parameter(torch.randn(r, b) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(a, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base: x @ W.T
        # lora: x @ A.T @ B.T * scaling   ==   ((x @ A.T) @ B.T) * scaling
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def delta_W(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


def demo_train_tiny() -> None:
    print("=" * 60)
    print("3) Tiny training loop: only A and B get gradients")
    print("=" * 60)

    torch.manual_seed(1)
    base = nn.Linear(32, 16, bias=False)
    layer = LoRALinear(base, r=4, alpha=8.0)

    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    frozen = [n for n, p in layer.named_parameters() if not p.requires_grad]
    print("trainable:", trainable)
    print("frozen:   ", frozen)

    opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(64, 32)
    # Teach the layer to map x -> 2*base(x) using only LoRA adapters
    with torch.no_grad():
        target = 2.0 * base(x)

    for step in range(50):
        opt.zero_grad()
        pred = layer(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 49:
            print(f"  step {step:02d}  loss={loss.item():.6f}  ||ΔW||={layer.delta_W().norm().item():.4f}")

    print()
    print("Learned ΔW shape:", tuple(layer.delta_W().shape), "(still a×b, but stored as B@A)")
    print()


if __name__ == "__main__":
    demo_shapes()
    demo_svd_intuition()
    demo_train_tiny()
    print("Done. Next: python 02_train_lora_qwen.py")
