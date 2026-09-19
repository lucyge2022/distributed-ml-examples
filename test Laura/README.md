# test Laura — LoRA on a small Qwen model

A hands-on demo of **LoRA (Low-Rank Adaptation)**: instead of updating a full weight matrix `W ∈ ℝ^{a×b}`, you learn two skinny factors whose product approximates the update.

```
ΔW ≈ B @ A
     │    └── A has shape (r, b)
     └─────── B has shape (a, r)

trainable params ≈ a·r + r·b   ≪   a·b   when r ≪ min(a, b)
```

That is the same low-rank idea as keeping the top singular directions of `ΔW` (SVD / "eigen" picture). Training does not run SVD each step — gradient descent just learns useful `A` and `B`.

This folder uses **Qwen2.5-0.5B-Instruct** as a stand-in for a much larger base model. The workflow is the same for 7B/70B: freeze the base, train adapters, ship only the small LoRA files.

## Files

| File | What it does |
|---|---|
| `01_lora_math.py` | Pure PyTorch: shapes, SVD intuition, tiny LoRA linear layer |
| `02_train_lora_qwen.py` | Download Qwen-0.5B, attach PEFT LoRA, train, save adapter, generate |
| `03_eval_base_vs_lora.py` | Benchmark base vs base+LoRA: PPL / NLL + side-by-side generations |
| `requirements.txt` | Dependencies |

## Setup

```bash
cd "test Laura"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run of `02_train_lora_qwen.py` downloads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face (~1 GB) into your HF cache.

## Run

```bash
# 1) math / shapes / SVD intuition (no download)
python 01_lora_math.py

# 2) real LoRA on Qwen-0.5B (downloads model, trains ~40 steps on CPU)
python 02_train_lora_qwen.py

# optional knobs
python 02_train_lora_qwen.py --rank 8 --alpha 16 --steps 40

# 3) compare base vs base+LoRA (PPL + side-by-side generations)
python 03_eval_base_vs_lora.py
```

`03_eval_base_vs_lora.py` writes `artifacts/eval_base_vs_lora.json`. On **general** held-out prompts you want small `|ΔNLL|` and high generation overlap — that means LoRA did not wreck the base model. On **LoRA-domain** prompts a light toy adapter may still look similar; a longer train would pull them apart.

Adapter weights land in `artifacts/qwen05b_lora/` (a few MB). Reload later with:

```bash
python 02_train_lora_qwen.py --skip-train
```

## How this maps to a big-model LoRA job

1. Load frozen base LLM (`AutoModelForCausalLM`).
2. Wrap with `LoraConfig` / `get_peft_model` targeting modules like `q_proj`, `v_proj`.
3. Optimize only `lora_A` / `lora_B` (and optionally biases you choose).
4. `save_pretrained` the adapter directory — not a full copy of the base.
5. At serve time: load base once, `PeftModel.from_pretrained(base, adapter_dir)`.

## Mental model check

For attention projection `W_q` of shape `(a, b)`:

- full fine-tune stores / updates `a×b` numbers
- LoRA stores `a×r` and `r×b` (your picture)
- forward uses `y = W x + scaling · B (A x)` so the big `W` stays frozen
