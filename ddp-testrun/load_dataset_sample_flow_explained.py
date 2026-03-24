"""
A sample flow of loading a dataset and using it in a model.

Binary file (on SSD)
    ↓ read bytes
In-memory array (NumPy)
  shape: (60000, 28, 28) uint8
    ↓ convert
tf.Tensor / torch.Tensor
  shape: (60000, 28, 28) float32
  normalized: divide by 255
  so values become 0.0 to 1.0
    ↓ batch
Mini-batch tensor
  shape: (32, 28, 28)  ← 32 images at a time
    ↓ flatten for simple model
  shape: (32, 784)     ← 28×28=784 pixels per image
    ↓ feed into model
Forward pass!

"""

import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

DATASET_FOLDER_PATH = "datasets/hojjatk/mnist-dataset/versions/1"
TRAIN_IMAGES_FILE = 'train-images.idx3-ubyte'
TRAIN_LABELS_FILE = 'train-labels.idx1-ubyte'

# Step 1: Read the binary file
def read_images(filepath):
    with open(filepath, 'rb') as f:
        # read 16-byte header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # read remaining bytes = raw pixels
        raw = f.read()
    return raw, num_images, rows, cols, magic

raw, num_images, rows, cols, magic = read_images(f'{DATASET_FOLDER_PATH}/{TRAIN_IMAGES_FILE}')
print(f"magic:      {magic}")   # 2051
print(f"num images: {num_images}")       # 60000
print(f"rows:       {rows}")    # 28
print(f"cols:       {cols}")    # 28
print(f"raw bytes:  {len(raw)}")# 47040000

# Step 2: Convert the raw bytes to a NumPy array
def convert_to_numpy(raw, num_objs, rows=0, cols=0):
    ndarray = np.frombuffer(raw, dtype=np.uint8)
    if rows and cols:
        return ndarray.reshape(num_objs, rows, cols)
    return ndarray
'''
np.frombuffer() reinterprets the raw bytes as numbers. 
reshape() gives them meaning: 60000 images, each 28 rows × 28 cols.
np.frombuffer() returns ndarray
    >>> np.frombuffer(b'\x01\x02', dtype=np.uint8)
    array([1, 2], dtype=uint8)
'''

images = convert_to_numpy(raw, num_images, rows, cols)
print(f"images shape: {images.shape}") # (60000, 28, 28)

# Step 3: Load the labels
def read_labels(filepath):
    with open(filepath, 'rb') as f:
        # read 8-byte header
        magic, num_labels = struct.unpack('>II', f.read(8))
        # read remaining bytes = raw labels
        raw = f.read()
    return raw, num_labels, magic

raw, num_labels, magic= read_labels(f'{DATASET_FOLDER_PATH}/{TRAIN_LABELS_FILE}')
print(f"magic:      {magic}")   # 2049
print(f"num labels: {num_labels}")       # 60000
labels = convert_to_numpy(raw, num_labels)
print(f"labels shape: {labels.shape}") # (60000,) this mean 60000 individual nums of data type uint8

# Step 4: Normalize the images
def normalize_images(images):
    return images.astype(np.float32) / 255.0

images_normalized = normalize_images(images)
print(f"images normalized shape: {images_normalized.shape}") # (60000, 28, 28)

# Step 5: Convert NumPy to PyTorch Tensor
# numpy array → pytorch tensor (still in CPU RAM)
X = torch.from_numpy(images_normalized)   # shape: (60000, 28, 28)  here no copy
y = torch.from_numpy(labels.astype(np.int64))  # shape: (60000,)
print(f"X.dtype:{X.dtype}") # torch.float32
print(f"y.dtype:{y.dtype}") # torch.int64
print(f"X.shape:{X.shape}") # torch.Size([60000, 28, 28])
print(f"y.shape:{y.shape}") # torch.Size([60000,])
# wrap in Dataset so DataLoader can batch it
dataset = TensorDataset(X, y)
# np.ndarray (60000,28,28) → torch.Tensor (60000,28,28)
print(f"dataset length: {len(dataset)}") # 60000

# Step 6 Batch with DataLoader into GPU
loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# training loop, lets just run one epoch, this part can't be run, refer to ddp_train.py for the full training loop
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# total_loss, steps = 0.0, 0
# for batch_images, batch_labels in loader:
#     # send mini batch from CPU RAM to GPU VRAM thru PCIe bus (32GB/s  usually around 10s of GB/s)
#     batch_images = batch_images.to(device) # shape [32,28,28] each batch is 32 * 28 * 28 * 4(float32) = 100k
#     batch_labels = batch_labels.to(device) # shape [32,]

#     # flatten 28×28 → 784 for simple model
#     batch_images = batch_images.view(32, -1)  # (32, 784)

#     optimizer.zero_grad()
#     loss = loss_fn(model(batch_images), batch_labels)
#     loss.backward()
#     optimizer.step()
#     total_loss += loss.item()
#     steps += 1



'''
## Full physical journey of your data
```
train-images.idx3-ubyte
        │
        │ open() + read()
        │ 3 GB/s
        ▼
CPU RAM: np.ndarray (60000,28,28)
        │
        │ torch.from_numpy()
        │ no copy! same RAM location
        ▼
CPU RAM: torch.Tensor (60000,28,28)
        │
        │ DataLoader slices batch of 32
        │ still in RAM
        ▼
CPU RAM: torch.Tensor (32,28,28)
        │
        │ .to('cuda')
        │ PCIe highway 32 GB/s
        ▼
GPU VRAM: torch.Tensor (32,28,28)
        │
        │ forward pass
        │ 2000 GB/s internal VRAM speed
        ▼
GPU VRAM: predictions, loss, gradients
```

---

## The beautiful connection
```
Hardware concept:          Code equivalent:
──────────────────────────────────────────
SSD                    =   .idx3-ubyte file
CPU RAM                =   numpy array / tensor before .to()
PCIe highway           =   .to('cuda') call
GPU VRAM               =   tensor after .to('cuda')
NVLink (multi-GPU)     =   .to('cuda:1'), .to('cuda:2') etc
'''