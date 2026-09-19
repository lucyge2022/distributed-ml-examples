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
| `04_visualize_eval.py` | Turn the eval JSON into PNG charts |
| `requirements.txt` | Dependencies |

---

## How to run this test and generate results

### 1. One-time setup

```bash
cd "test Laura"
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first train run downloads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face (~1 GB) into your HF cache.

### 2. End-to-end pipeline (train → eval → plots)

```bash
# A) optional: understand the ΔW ≈ B@A math (no model download)
python 01_lora_math.py

# B) train a tiny LoRA adapter on Qwen-0.5B  →  artifacts/qwen05b_lora/
python 02_train_lora_qwen.py
# optional knobs:
# python 02_train_lora_qwen.py --rank 8 --alpha 16 --steps 40

# C) evaluate base vs base+LoRA               →  artifacts/eval_base_vs_lora.json
python 03_eval_base_vs_lora.py

# D) visualize the eval JSON                  →  artifacts/plots/*.png
python 04_visualize_eval.py
```

Open the PNGs under `artifacts/plots/` (or the JSON if you want raw numbers).

### 3. What each artifact means

| Output | Meaning |
|---|---|
| `artifacts/qwen05b_lora/` | Saved LoRA adapter only (~2 MB weights), not a full model copy |
| `artifacts/eval_base_vs_lora.json` | Per-prompt NLL/PPL, generations, Jaccard, exact-match |
| `artifacts/plots/summary_base_vs_lora.png` | Mean PPL / Jaccard / exact-match by split |
| `artifacts/plots/delta_nll_summary.png` | How much LoRA shifted likelihood (`ΔNLL ≈ 0` ⇒ similar) |
| `artifacts/plots/nll_by_prompt_*.png` | Per-prompt base vs LoRA NLL bars |
| `artifacts/plots/jaccard_by_prompt_*.png` | Per-prompt generation overlap |

**How to read “more or less the same”**

- On **GENERAL** prompts: small `|mean ΔNLL|` (roughly &lt; 0.15) and higher Jaccard/exact-match ⇒ LoRA did not wreck the base.
- On **LORA-DOMAIN** prompts: larger ΔNLL / lower overlap is OK — that is the adapter doing work.

### 4. Re-run pieces independently

```bash
# reload existing adapter and generate only
python 02_train_lora_qwen.py --skip-train

# re-eval without retraining (needs artifacts/qwen05b_lora/)
python 03_eval_base_vs_lora.py

# re-plot without re-eval (needs artifacts/eval_base_vs_lora.json)
python 04_visualize_eval.py
```

---

## How this maps to a big-model LoRA job

1. Load frozen base LLM (`AutoModelForCausalLM`).
2. Wrap with `LoraConfig` / `get_peft_model` targeting modules like `q_proj`, `v_proj`.
3. Optimize only `lora_A` / `lora_B` (and optionally biases you choose).
4. `save_pretrained` the adapter directory — not a full copy of the base.
5. At serve time: load base once, `PeftModel.from_pretrained(base, adapter_dir)`.
6. Eval + plot the same way: compare base vs base+adapter on held-out prompts.

## Mental model check

For attention projection `W_q` of shape `(a, b)`:

- full fine-tune stores / updates `a×b` numbers
- LoRA stores `a×r` and `r×b` (your picture)
- forward uses `y = W x + scaling · B (A x)` so the big `W` stays frozen
