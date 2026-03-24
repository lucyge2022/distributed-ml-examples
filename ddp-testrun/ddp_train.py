"""
Simple DDP (DistributedDataParallel) training demo using mp.spawn.
Use AllReduce to sync gradients across all processes.

Launch with:
    python ddp_train.py            # defaults to 2 workers
    python ddp_train.py --nproc 4


"""

import argparse
import os, time
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def setup(rank: int, world_size: int, init_file: str):
    # File-based rendezvous — no TCP store, no IPv6 DNS issues on macOS
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )


def train(rank: int, world_size: int, init_file: str):
    setup(rank, world_size, init_file)

    print(f"rank={rank} pid={os.getpid()} ppid={os.getppid()}")
    dist.barrier()
    if rank == 0:
        print(f"Starting DDP training — world_size={world_size}")

    # Synthetic dataset: 1000 samples, 20 features, binary labels
    torch.manual_seed(42)
    X = torch.randn(1000, 20) # 1000 samples, 20 features, a 1000x20 tensor
    y = torch.randint(0, 2, (1000,)) # binary labels, 0 or 1, a 1000x1 tensor, labeling the 1000 samples as YES or NO for some conceptual categorization
    dataset = TensorDataset(X, y) # already tensor-ized dataset, ready to be fed into model

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = build_model()
    model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)  # different shuffle per epoch across ranks
        total_loss, steps = 0.0, 0

        for xb, yb in loader: # batch of 32 samples(each with 20 features and a binary label), xb is a batch of training data, yb is the batch of labels
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        time.sleep(1)
        if rank == 0:
            print(f"Epoch {epoch + 1}/5  avg_loss={total_loss / steps:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")
        print("Checkpoint saved to checkpoint.pt")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=2, help="Number of worker processes")
    args = parser.parse_args()

    # Temp file for rendezvous — deleted after run
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ddp_rendezvous") as f:
        init_file = f.name
    os.unlink(init_file)  # file must not exist before init_process_group

    try:
        mp.spawn(
            train,
            args=(args.nproc, init_file),
            nprocs=args.nproc,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


if __name__ == "__main__":
    main()
