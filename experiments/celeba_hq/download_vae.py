"""Run LOCALLY to fetch the pretrained SD VAE used to compress CelebA-HQ into
a latent diffusion space (Rombach et al. "High-Resolution Image Synthesis with
Latent Diffusion Models", CVPR 2022). No login/token needed, unlike IEEE
DataPort -- this is what makes the celeba_hq pipeline scriptable end to end
except for the raw image download itself.

stabilityai/sd-vae-ft-mse: SD's original VAE, finetuned by StabilityAI for
better reconstruction MSE/LPIPS. 8x spatial downsample, 4 latent channels --
a 256x256 image becomes a 32x32x4 latent, which is what makes training a
diffusion model on 30k CelebA-HQ images tractable without a GPU cluster the
size of the one Rombach et al. used.
"""
import os
from huggingface_hub import snapshot_download

REPO_ID = "stabilityai/sd-vae-ft-mse"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "pretrained", "sd-vae-ft-mse")

if __name__ == "__main__":
    path = snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    print(f"downloaded {REPO_ID} to {path}")
