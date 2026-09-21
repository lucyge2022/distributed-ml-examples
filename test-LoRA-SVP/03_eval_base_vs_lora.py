"""
Eval: base Qwen vs base + LoRA — are they "more or less the same"?

After a light LoRA train, the frozen base should still dominate general
behavior. This script measures that with:

  1. Causal LM loss / perplexity on held-out prompts (general + LoRA-domain)
  2. Side-by-side greedy generations
  3. Token overlap / exact-match rate between the two models' outputs

Expectation for this toy adapter (few steps, tiny corpus):
  - general prompts: base and base+LoRA stay close (similar PPL, similar text)
  - LoRA-domain prompts: may diverge a bit toward the toy answers

Requires a trained adapter from 02_train_lora_qwen.py:
    python 02_train_lora_qwen.py
    python 03_eval_base_vs_lora.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent / "artifacts" / "qwen05b_lora"
REPORT_PATH = Path(__file__).resolve().parent / "artifacts" / "eval_base_vs_lora.json"

# Held-out general knowledge / chat — should stay similar if LoRA is light.
GENERAL_PROMPTS = [
    "User: What is the capital of France?\nAssistant:",
    "User: Explain gravity in one sentence.\nAssistant:",
    "User: Write a haiku about the ocean.\nAssistant:",
    "User: What is 17 + 25?\nAssistant:",
    "User: Name three primary colors.\nAssistant:",
]

# Same style as the toy train set, but not the exact train strings.
LORA_DOMAIN_PROMPTS = [
    "User: What is LoRA used for?\nAssistant:",
    "User: Why use low-rank matrices when fine-tuning?\nAssistant:",
    "User: How do you ship a LoRA adapter?\nAssistant:",
    "User: What is the LoRA rank r?\nAssistant:",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base model vs base+LoRA")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    return p.parse_args()


def load_models(model_id: str, adapter: Path):
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(
            f"No adapter at {adapter}. Train first:\n"
            f"  python 02_train_lora_qwen.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading base: {model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True
    )
    base.eval()

    print(f"loading base + LoRA: {adapter}")
    # Separate copy so we can score both without enable/disable races.
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True
    )
    lora = PeftModel.from_pretrained(base_for_lora, adapter)
    lora.eval()
    return tokenizer, base, lora


@torch.no_grad()
def sequence_nll(model, tokenizer, prompt: str, continuation: str | None = None) -> dict:
    """
    If continuation is None, score teacher-forced NLL of the prompt itself
    (next-token prediction over the prompt tokens).
    If continuation is given, score NLL of continuation tokens given the prompt.
    """
    if continuation is None:
        ids = tokenizer(prompt, return_tensors="pt")
        input_ids = ids["input_ids"]
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()
        n_tokens = input_ids.size(1) - 1
        return {"nll": loss, "ppl": float(torch.exp(torch.tensor(loss))), "tokens": n_tokens}

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    labels = labels[: len(full_ids)]
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    label_t = torch.tensor([labels], dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=label_t)
    loss = outputs.loss.item()
    n_tokens = sum(1 for t in labels if t != -100)
    return {"nll": loss, "ppl": float(torch.exp(torch.tensor(loss))), "tokens": n_tokens}


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the completion after the prompt when possible.
    if full.startswith(prompt):
        return full[len(prompt) :].strip()
    return full.strip()


def token_overlap(a: str, b: str) -> float:
    ta = a.lower().split()
    tb = b.lower().split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / len(sa | sb)


def eval_split(name: str, prompts: list[str], tokenizer, base, lora, max_new_tokens: int) -> dict:
    print("\n" + "=" * 72)
    print(f"split: {name}  ({len(prompts)} prompts)")
    print("=" * 72)

    rows = []
    base_nlls, lora_nlls, overlaps = [], [], []

    for i, prompt in enumerate(prompts, 1):
        b_score = sequence_nll(base, tokenizer, prompt)
        l_score = sequence_nll(lora, tokenizer, prompt)
        b_gen = generate(base, tokenizer, prompt, max_new_tokens)
        l_gen = generate(lora, tokenizer, prompt, max_new_tokens)
        overlap = token_overlap(b_gen, l_gen)
        exact = b_gen == l_gen

        base_nlls.append(b_score["nll"])
        lora_nlls.append(l_score["nll"])
        overlaps.append(overlap)

        row = {
            "prompt": prompt,
            "base_nll": b_score["nll"],
            "lora_nll": l_score["nll"],
            "base_ppl": b_score["ppl"],
            "lora_ppl": l_score["ppl"],
            "nll_delta": l_score["nll"] - b_score["nll"],
            "base_gen": b_gen,
            "lora_gen": l_gen,
            "token_jaccard": overlap,
            "exact_match": exact,
        }
        rows.append(row)

        print(f"\n[{i}/{len(prompts)}] {prompt.splitlines()[0]}")
        print(f"  base  NLL={b_score['nll']:.4f}  PPL={b_score['ppl']:.2f}")
        print(f"  lora  NLL={l_score['nll']:.4f}  PPL={l_score['ppl']:.2f}  "
              f"ΔNLL={row['nll_delta']:+.4f}")
        print(f"  gen overlap (Jaccard)={overlap:.3f}  exact={exact}")
        print(f"  BASE>  {b_gen[:160]}{'...' if len(b_gen) > 160 else ''}")
        print(f"  LORA>  {l_gen[:160]}{'...' if len(l_gen) > 160 else ''}")

    summary = {
        "n_prompts": len(prompts),
        "mean_base_nll": sum(base_nlls) / len(base_nlls),
        "mean_lora_nll": sum(lora_nlls) / len(lora_nlls),
        "mean_nll_delta": (sum(lora_nlls) - sum(base_nlls)) / len(base_nlls),
        "mean_base_ppl": float(torch.exp(torch.tensor(sum(base_nlls) / len(base_nlls)))),
        "mean_lora_ppl": float(torch.exp(torch.tensor(sum(lora_nlls) / len(lora_nlls)))),
        "mean_token_jaccard": sum(overlaps) / len(overlaps),
        "exact_match_rate": sum(1 for r in rows if r["exact_match"]) / len(rows),
    }
    print("\n--- summary ---")
    print(f"  mean base PPL : {summary['mean_base_ppl']:.3f}")
    print(f"  mean lora PPL : {summary['mean_lora_ppl']:.3f}")
    print(f"  mean ΔNLL     : {summary['mean_nll_delta']:+.4f}  "
          f"(~0 means LoRA barely changed likelihood on these prompts)")
    print(f"  mean Jaccard  : {summary['mean_token_jaccard']:.3f}")
    print(f"  exact match   : {summary['exact_match_rate']:.0%}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    args = parse_args()
    t0 = time.time()
    tokenizer, base, lora = load_models(args.model, args.adapter)

    report = {
        "base_model": args.model,
        "adapter": str(args.adapter),
        "max_new_tokens": args.max_new_tokens,
        "note": (
            "Small |ΔNLL| and high generation overlap on GENERAL means "
            "LoRA preserved base behavior. LoRA-domain may diverge after training."
        ),
        "general": eval_split(
            "GENERAL (should stay similar)",
            GENERAL_PROMPTS,
            tokenizer,
            base,
            lora,
            args.max_new_tokens,
        ),
        "lora_domain": eval_split(
            "LORA-DOMAIN (may diverge a little)",
            LORA_DOMAIN_PROMPTS,
            tokenizer,
            base,
            lora,
            args.max_new_tokens,
        ),
        "elapsed_sec": round(time.time() - t0, 2),
    }

    # One-line verdict
    g = report["general"]["summary"]
    print("\n" + "#" * 72)
    print("VERDICT (general held-out prompts)")
    print("#" * 72)
    close_nll = abs(g["mean_nll_delta"]) < 0.15
    close_gen = g["mean_token_jaccard"] >= 0.5 or g["exact_match_rate"] >= 0.4
    if close_nll and close_gen:
        print("Base and base+LoRA look MORE OR LESS THE SAME on general prompts.")
    elif close_nll:
        print("Likelihoods are close; generations differ more (still a light adapter).")
    else:
        print("Noticeable likelihood shift — adapter moved the model; inspect rows.")
    print(
        f"|ΔNLL|={abs(g['mean_nll_delta']):.4f}  "
        f"Jaccard={g['mean_token_jaccard']:.3f}  "
        f"exact={g['exact_match_rate']:.0%}"
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote report -> {args.report}")


if __name__ == "__main__":
    main()
