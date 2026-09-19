"""
Train a LoRA adapter on a small Qwen model (Qwen2.5-0.5B-Instruct).

What this shows for a "big" model workflow:
  1. Download / load a frozen base LLM.
  2. Attach LoRA adapters (two skinny matrices per target linear layer).
  3. Train only those adapters on a tiny toy corpus.
  4. Save the LoRA weights alone (MBs, not GBs).
  5. Reload base + LoRA and run a short generation.

Your mental model, mapped to PEFT naming for a weight W of shape (a, b)
= (out_features, in_features):

    lora_A : (r, b)     # the r×b factor
    lora_B : (a, r)     # the a×r factor
    ΔW     = scaling * (B @ A)     # reconstructs an a×b update

Run (from this folder, with the local venv active):
    python 02_train_lora_qwen.py
    python 02_train_lora_qwen.py --steps 30 --rank 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "qwen05b_lora"


TOY_PROMPTS = [
    ("User: What is LoRA?\nAssistant:", "LoRA is low-rank adaptation for fine-tuning large models."),
    ("User: Why freeze the base weights?\nAssistant:", "Freezing keeps the big model intact and trains only tiny adapters."),
    ("User: What shapes do LoRA matrices have?\nAssistant:", "For a weight a by b, LoRA uses a by r and r by b matrices."),
    ("User: What is rank r?\nAssistant:", "Rank r is the bottleneck width of the low-rank update."),
    ("User: Can I ship only the adapter?\nAssistant:", "Yes, save the LoRA weights and reload them on top of the base model."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny LoRA training demo on Qwen2.5-0.5B")
    p.add_argument("--model", default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--rank", type=int, default=8, help="LoRA rank r")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/r)")
    p.add_argument("--steps", type=int, default=40, help="Optimizer steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--skip-train", action="store_true", help="Only load + generate if adapter exists")
    return p.parse_args()


def build_batch(tokenizer, device: torch.device):
    """Pack toy prompt/answer pairs into one padded causal-LM batch."""
    input_ids = []
    labels = []
    for prompt, answer in TOY_PROMPTS:
        full = prompt + " " + answer
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        # Causal LM: predict every next token; mask the prompt portion in the loss.
        lab = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        # Keep label length aligned with input
        if len(lab) < len(full_ids):
            lab = lab + [-100] * (len(full_ids) - len(lab))
        lab = lab[: len(full_ids)]
        input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))

    max_len = max(t.size(0) for t in input_ids)
    pad_id = tokenizer.pad_token_id

    def pad(seq: torch.Tensor, value: int) -> torch.Tensor:
        if seq.size(0) == max_len:
            return seq
        return torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=value)

    ids = torch.stack([pad(t, pad_id) for t in input_ids]).to(device)
    labs = torch.stack([pad(t, -100) for t in labels]).to(device)
    attn = (ids != pad_id).long()
    return {"input_ids": ids, "attention_mask": attn, "labels": labs}


def print_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")


def show_one_lora_shapes(model) -> None:
    """Print A/B shapes for the first LoRA layer we find — your a×r and r×b picture."""
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # peft stores adapters in ModuleDict keyed by adapter name, usually "default"
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            print(f"example adapter on '{name}':")
            print(f"  lora_A (r, b) = {tuple(A.shape)}")
            print(f"  lora_B (a, r) = {tuple(B.shape)}")
            print(f"  implied ΔW    = B@A -> ({B.shape[0]}, {A.shape[1]})")
            return
    print("no LoRA modules found to inspect")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    print(f"loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        # Attention projections are the usual place to attach adapters.
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.train()

    print_trainable(model)
    show_one_lora_shapes(model)

    batch = build_batch(tokenizer, device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    print(f"\ntraining for {args.steps} steps on {len(TOY_PROMPTS)} toy examples...")
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"  step {step:03d}  loss={loss.item():.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    meta = {
        "base_model": args.model,
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": args.steps,
        "target_modules": ["q_proj", "v_proj"],
        "note": "Adapter only — reload with PeftModel.from_pretrained(base, this_dir)",
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved LoRA adapter to {args.out}")
    # Rough on-disk size of adapter weights
    adapter_files = list(args.out.glob("adapter_model.*")) + list(args.out.glob("*.safetensors"))
    if adapter_files:
        size_mb = sum(f.stat().st_size for f in set(adapter_files)) / (1024 * 1024)
        print(f"adapter weight files ≈ {size_mb:.2f} MB (base model stays frozen elsewhere)")


@torch.no_grad()
def generate_demo(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    print("\n" + "=" * 60)
    print("generation with base + LoRA")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.out, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.out)
    model.to(device)
    model.eval()

    prompt = "User: What shapes do LoRA matrices have?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


def main() -> None:
    args = parse_args()
    if args.skip_train:
        if not (args.out / "adapter_config.json").exists():
            raise SystemExit(f"no adapter at {args.out}; run without --skip-train first")
    else:
        train(args)
    generate_demo(args)


if __name__ == "__main__":
    main()
