"""Encodes all CelebA-HQ images through the frozen SD VAE once and caches the latents to a
single tensor file: faster (VAE encode dominates wall-clock if repeated every training step)
and smaller (30k images at 4x32x32 fp16 is ~16MB, fits in memory with no DataLoader overhead).

Horizontal flip is baked in as a fixed augmentation (each image encoded twice) rather than
done randomly at train time, since flipping post-hoc in latent space isn't equivalent to
flipping the pixel image before the VAE.

Latents are scaled by the VAE's own `scaling_factor` (Rombach et al. 2022) so a single,
dataset-agnostic EDM sigma schedule works without hand-retuning sigma_data.
"""
import os
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from celeba_hq_data import CelebAHQImages

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
