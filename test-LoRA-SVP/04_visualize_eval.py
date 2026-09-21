"""
Visualize base vs LoRA eval results from 03_eval_base_vs_lora.py.

Reads artifacts/eval_base_vs_lora.json and writes PNG charts:
  - per-prompt NLL (base vs LoRA)
  - per-prompt generation Jaccard overlap
  - summary bars (mean PPL, mean Jaccard, exact-match rate)

Run:
    python 03_eval_base_vs_lora.py          # produce the JSON first
    python 04_visualize_eval.py             # write PNGs under artifacts/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REPORT = Path(__file__).resolve().parent / "artifacts" / "eval_base_vs_lora.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts" / "plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot base vs LoRA eval JSON")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def short_label(prompt: str, max_len: int = 28) -> str:
    line = prompt.splitlines()[0]
    line = line.replace("User: ", "").strip()
    if len(line) > max_len:
        return line[: max_len - 1] + "…"
    return line


def plot_nll_by_prompt(split_name: str, rows: list[dict], out: Path) -> Path:
    labels = [short_label(r["prompt"]) for r in rows]
    base = [r["base_nll"] for r in rows]
    lora = [r["lora_nll"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, base, width, label="base", color="#3d5a80")
    ax.bar(x + width / 2, lora, width, label="base+LoRA", color="#ee6c4d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("NLL (lower = model finds prompt more likely)")
    ax.set_title(f"Base vs LoRA — NLL by prompt ({split_name})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / f"nll_by_prompt_{split_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_jaccard_by_prompt(split_name: str, rows: list[dict], out: Path) -> Path:
    labels = [short_label(r["prompt"]) for r in rows]
    jacc = [r["token_jaccard"] for r in rows]
    colors = ["#2a9d8f" if r["exact_match"] else "#e9c46a" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, jacc, color=colors)
    ax.axhline(0.5, color="#264653", linestyle="--", linewidth=1, label="0.5 overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Token Jaccard (base gen vs LoRA gen)")
    ax.set_title(f"Generation overlap ({split_name}) — green = exact match")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / f"jaccard_by_prompt_{split_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary(report: dict, out: Path) -> Path:
    splits = ["general", "lora_domain"]
    titles = ["GENERAL", "LORA-DOMAIN"]
    metrics = [
        ("mean_base_ppl", "mean_lora_ppl", "Mean PPL"),
        ("mean_token_jaccard", "mean_token_jaccard", "Mean Jaccard"),
        ("exact_match_rate", "exact_match_rate", "Exact-match rate"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    # PPL grouped bars
    ax = axes[0]
    x = np.arange(len(splits))
    width = 0.35
    base_ppl = [report[s]["summary"]["mean_base_ppl"] for s in splits]
    lora_ppl = [report[s]["summary"]["mean_lora_ppl"] for s in splits]
    ax.bar(x - width / 2, base_ppl, width, label="base", color="#3d5a80")
    ax.bar(x + width / 2, lora_ppl, width, label="base+LoRA", color="#ee6c4d")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("Perplexity")
    ax.set_title("Mean PPL by split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Jaccard
    ax = axes[1]
    jacc = [report[s]["summary"]["mean_token_jaccard"] for s in splits]
    ax.bar(titles, jacc, color=["#2a9d8f", "#e76f51"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard")
    ax.set_title("Mean generation overlap")
    ax.grid(axis="y", alpha=0.3)

    # Exact match
    ax = axes[2]
    exact = [report[s]["summary"]["exact_match_rate"] for s in splits]
    ax.bar(titles, exact, color=["#2a9d8f", "#e76f51"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Exact-match rate")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Base vs base+LoRA — eval summary", fontsize=13)
    fig.tight_layout()
    path = out / "summary_base_vs_lora.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delta_nll(report: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    splits = ["general", "lora_domain"]
    titles = ["GENERAL\n(should stay close)", "LORA-DOMAIN\n(may diverge)"]
    deltas = [report[s]["summary"]["mean_nll_delta"] for s in splits]
    colors = ["#2a9d8f" if abs(d) < 0.15 else "#e76f51" for d in deltas]
    ax.bar(titles, deltas, color=colors)
    ax.axhline(0, color="#264653", linewidth=1)
    ax.axhline(0.15, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(-0.15, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("mean ΔNLL  (LoRA − base)")
    ax.set_title("Likelihood shift — dashed band ≈ “more or less the same”")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "delta_nll_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if not args.report.exists():
        raise SystemExit(
            f"No report at {args.report}. Run eval first:\n"
            f"  python 03_eval_base_vs_lora.py"
        )

    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.append(plot_summary(report, args.out))
    written.append(plot_delta_nll(report, args.out))
    for split in ("general", "lora_domain"):
        rows = report[split]["rows"]
        written.append(plot_nll_by_prompt(split, rows, args.out))
        written.append(plot_jaccard_by_prompt(split, rows, args.out))

    print(f"read  {args.report}")
    print(f"wrote {len(written)} plots -> {args.out}")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
