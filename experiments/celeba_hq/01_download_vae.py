"""Download the pretrained SD VAE (stabilityai/sd-vae-ft-mse; 8x downsample, 4 latent channels)."""
import os
from huggingface_hub import snapshot_download

REPO_ID = "stabilityai/sd-vae-ft-mse"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "pretrained", "sd-vae-ft-mse")

if __name__ == "__main__":
    path = snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    print(f"downloaded {REPO_ID} to {path}")
