# Distributed ML Examples

Runnable code examples that accompany the [Distributed ML Notes](https://github.com/lucyge2022/Distributed-ML-Notes) learning series.

Each subdirectory is a self-contained toy program with its own `README.md` and `requirements.txt`.

---

## Examples

| Directory | What it covers | Related notes chapter |
|---|---|---|
| [`ddp-testrun`](./ddp-testrun/) | PyTorch DDP training with Ring AllReduce on MNIST | [Chapter 3 — Worker-Only (AllReduce)](https://lucyge2022.github.io/Distributed-ML-Notes/chapter-3/AllReduce-Pattern.html) |
| [`vllm-test`](./vllm-test/) | vLLM serving: prefill vs decode, static vs continuous batching, TTFT, KV cache / PagedAttention | [Prefill vs Decode](https://github.com/lucyge2022/Distributed-ML-Notes/blob/main/src/LLM%20inference/Prefill%20%26%20Decode.md) |
| [`test Laura`](./test%20Laura/) | LoRA math (`ΔW ≈ B@A`) + PEFT LoRA on Qwen2.5-0.5B | — |

---

## How to use

Each example is standalone. Navigate into the subdirectory and follow its README:

```bash
cd ddp-testrun
pip install -r requirements.txt
python ddp_train.py
```

---

## Planned examples

- `allreduce-from-scratch` — Ring AllReduce implemented manually without DDP
- `dataset-loading` — Distributed dataset sharding and DataLoader setup
