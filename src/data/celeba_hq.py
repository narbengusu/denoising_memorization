"""Lazy CelebA-HQ 256x256 dataset. Download the Kaggle mirror badasstechie/celebahq-resized-256x256 and put the 30k
.jpg files in data/celeba_hq/images/."""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DATA_DIR = os.environ.get("CELEBA_HQ_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "celeba_hq"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")


class CelebAHQImages(Dataset):
    """float32 CHW tensors in [-1, 1], as the SD VAE encoder expects."""

    def __init__(self, resolution=256, flip=False):
        if not os.path.isdir(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
            raise FileNotFoundError(
                f"{IMAGES_DIR} not found or empty. Download the 256x256 CelebA-HQ images "
                "(e.g. kaggle datasets download -d badasstechie/celebahq-resized-256x256) "
                f"and place the .jpg files at {IMAGES_DIR}/."
            )
        self.dir = IMAGES_DIR
        self.files = sorted(os.listdir(self.dir))
        self.resolution = resolution
        self.flip = flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.files[idx])).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        return x / 127.5 - 1.0
