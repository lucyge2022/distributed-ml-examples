# Distributed ML Examples

Runnable code examples that accompany the [Distributed ML Notes](https://github.com/lucyge/Distributed-ML-Notes) learning series.

Each subdirectory is a self-contained toy program with its own `README.md` and `requirements.txt`.

---

## Examples

| Directory | What it covers | Related notes chapter |
|---|---|---|
| [`ddp-testrun`](./ddp-testrun/) | PyTorch DDP training with Ring AllReduce on MNIST | [Chapter 3 — Worker-Only (AllReduce)](https://github.com/lucyge/Distributed-ML-Notes) |

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

- `ps-pattern-sim` — Parameter Server + Workers pattern simulation
- `allreduce-from-scratch` — Ring AllReduce implemented manually without DDP
- `dataset-loading` — Distributed dataset sharding and DataLoader setup
