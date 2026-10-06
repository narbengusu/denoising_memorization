"""Encode CelebA-HQ and its horizontal flips through the frozen SD VAE once and cache the scaled latents.
Flips are applied in pixel space, since flipping latents is not equivalent."""
import os
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from data.celeba_hq import CelebAHQImages

device = os.environ.get("DEVICE", "cpu")
vae_dir = os.environ.get("VAE_DIR", "pretrained/sd-vae-ft-mse")
resolution = int(os.environ.get("RESOLUTION", 256))
batch = int(os.environ.get("BATCH", 32))
out_path = os.environ.get("OUT_PATH", "checkpoints/celeba_hq_latents.pt")

os.makedirs("checkpoints", exist_ok=True)
vae = AutoencoderKL.from_pretrained(vae_dir).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
scaling_factor = vae.config.scaling_factor

all_latents = []
with torch.no_grad():
    for flip in (False, True):
        ds = CelebAHQImages(resolution=resolution, flip=flip)
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
        for i, x in enumerate(dl):
            x = x.to(device)
            posterior = vae.encode(x).latent_dist
            z = posterior.sample() * scaling_factor
            all_latents.append(z.half().cpu())
            if i % 50 == 0:
                print(f"flip={flip} batch {i}/{len(dl)}", flush=True)

latents = torch.cat(all_latents, dim=0)
sigma_data = latents.float().std().item()
print(f"encoded {latents.shape[0]} latents, shape {tuple(latents.shape[1:])}, "
      f"empirical sigma_data={sigma_data:.4f}", flush=True)

torch.save({"latents": latents, "sigma_data": sigma_data, "scaling_factor": scaling_factor,
            "resolution": resolution}, out_path)
print(f"saved to {out_path}", flush=True)
