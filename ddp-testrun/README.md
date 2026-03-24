# DDP Testrun

A minimal local distributed training demo using **PyTorch DDP (DistributedDataParallel)**.

## What it does

Simulates multi-GPU distributed training on a single machine by spawning N worker processes. Each process:

1. Gets its own shard of the dataset via `DistributedSampler`
2. Trains independently on its shard
3. Automatically syncs gradients across all processes via DDP allreduce after each backward pass

Uses a 2-layer MLP on synthetic data (1000 samples, 20 features, binary classification). Only rank 0 prints logs and saves the checkpoint.

**Why `mp.spawn` instead of `torchrun`**: On macOS, `torchrun --standalone` uses a TCP store that binds to the machine hostname, which resolves to IPv6 `::1`. The `gloo` backend then hangs indefinitely doing reverse IPv6 DNS lookups. Using `mp.spawn` with a file-based rendezvous (`file://...`) bypasses this entirely.

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that keeps dependencies isolated from your system Python.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv .venv
uv pip install -r requirements.txt

# Activate the venv
source .venv/bin/activate
```

### With pip + venv (standard)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Make sure the venv is activated (`source .venv/bin/activate`), then:

```bash
# 2 worker processes (default)
python3 ddp_train.py

# 4 worker processes
python3 ddp_train.py --nproc 4
```

Or run directly without activating:

```bash
.venv/bin/python3 ddp_train.py --nproc 2
```

## Expected output

```
Starting DDP training — world_size=2
Epoch 1/5  avg_loss=0.7083
Epoch 2/5  avg_loss=0.6922
Epoch 3/5  avg_loss=0.6904
Epoch 4/5  avg_loss=0.6803
Epoch 5/5  avg_loss=0.6884
Checkpoint saved to checkpoint.pt
```

After training, `checkpoint.pt` is written to the current directory containing the model weights.

## Key concepts demonstrated

| Concept | Where |
|---|---|
| `dist.init_process_group` | Sets up the process group with `gloo` backend |
| `DistributedSampler` | Partitions dataset across ranks, no overlap |
| `DDP(model)` | Wraps model so gradients sync automatically |
| `sampler.set_epoch(epoch)` | Ensures different shuffle order each epoch |
| Rank 0 only logging/saving | `if rank == 0:` guards around print and `torch.save` |

## Extending this

- **Real dataset**: swap `TensorDataset` for `torchvision.datasets.MNIST`
- **Apple Silicon (MPS)**: move tensors to `torch.device("mps")` and set `map_location` in `torch.load`
- **Multi-node**: replace the file rendezvous with `init_method="tcp://MASTER_IP:PORT"` and set `RANK`/`WORLD_SIZE` env vars per node
- **Larger models**: add gradient accumulation or `torch.cuda.amp` for mixed precision
