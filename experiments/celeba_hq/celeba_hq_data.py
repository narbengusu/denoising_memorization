"""CelebA-HQ has no scriptable download: the IEEE DataPort listing
(ieee-dataport.org/documents/celeba-hq) sits behind a paid-subscription login
wall with no API. Used the Kaggle mirror instead (badasstechie/celebahq-
resized-256x256, free account + API token, no subscription) -- it's already
downscaled to 256x256, so unlike the original 1024x1024 IEEE release there's
no separate extract/resize step: place the 30k .jpg files directly at
data/celeba_hq/images/ (repo-root data/, same convention cifar10_data.py uses).

Loading is still lazy (PIL open per __getitem__, not eager like
cifar10_data.py) even though 30k x 256x256x3 float32 would only be ~6GB --
DataLoader workers overlap decode with GPU compute for free, so there's no
reason to pay the eager-load memory cost up front.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DATA_DIR = os.environ.get("CELEBA_HQ_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "celeba_hq"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")


class CelebAHQImages(Dataset):
    """Returns float32 tensors in [-1, 1], CHW -- matches what the SD VAE
    encoder expects. Images are already 256x256; `resolution` only matters
    if a differently-sized source is swapped in later."""

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
