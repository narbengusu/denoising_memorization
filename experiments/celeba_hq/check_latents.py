"""Sanity check for checkpoints/celeba_hq_latents.pt before committing to the
multi-day train_celeba_hq_edm.py run -- decodes a handful of cached latents
back through the VAE so a scaling/encoding bug shows up as visibly wrong
faces now, not as a bad loss curve two days into training.
"""
import os
import torch
import torchvision.utils as vutils
from diffusers import AutoencoderKL

device = os.environ.get("DEVICE", "cpu")
vae_dir = os.environ.get("VAE_DIR", "pretrained/sd-vae-ft-mse")
latents_path = os.environ.get("LATENTS_PATH", "checkpoints/celeba_hq_latents.pt")
out_path = os.environ.get("OUT_PATH", "checkpoints/latents_check_grid.png")
n_check = int(os.environ.get("N_CHECK", 16))

blob = torch.load(latents_path, map_location="cpu")
latents, sigma_data, scaling_factor = blob["latents"], blob["sigma_data"], blob["scaling_factor"]
print(f"latents shape={tuple(latents.shape)} dtype={latents.dtype} "
      f"mean={latents.float().mean():.4f} std={latents.float().std():.4f} sigma_data={sigma_data:.4f}")
assert latents.ndim == 4 and latents.shape[1:] == (4, 32, 32), "unexpected latent shape"
assert torch.isfinite(latents.float()).all(), "latents contain NaN/inf"

idx = torch.randperm(latents.shape[0])[:n_check]
z = latents[idx].float().to(device) / scaling_factor

vae = AutoencoderKL.from_pretrained(vae_dir).to(device).eval()
with torch.no_grad():
    images = vae.decode(z).sample.clamp(-1, 1)

vutils.save_image(images, out_path, nrow=4, normalize=True, value_range=(-1, 1))
print(f"saved {out_path} -- inspect it: should look like real faces, not noise")
