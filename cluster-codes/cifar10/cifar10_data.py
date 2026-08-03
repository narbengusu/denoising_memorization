"""Loads the raw CIFAR-10 pickled batches (no torchvision dependency) and
returns [-1, 1]-scaled float32 tensors in NCHW, matching what
diffusers.UNet2DModel / DDPMScheduler expect."""
import os
import pickle
import numpy as np
import torch

DATA_DIR = os.environ.get("CIFAR10_DATA_DIR", "../data/cifar-10-batches-py")


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])


def load_cifar10(data_dir=DATA_DIR, split="train"):
    """Returns (images [N,3,32,32] float32 in [-1,1], labels [N] int64)."""
    if split == "train":
        batches = [f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        batches = ["test_batch"]
    else:
        raise ValueError(split)

    data_parts, label_parts = [], []
    for b in batches:
        data, labels = _load_batch(os.path.join(data_dir, b))
        data_parts.append(data)
        label_parts.append(labels)
    data = np.concatenate(data_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)

    images = data.reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(images), torch.from_numpy(labels)
