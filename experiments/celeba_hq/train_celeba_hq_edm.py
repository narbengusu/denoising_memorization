"""Trains a latent diffusion model on precomputed CelebA-HQ VAE latents
(run precompute_latents.py first). Three tricks on top of a plain VP-SDE noise-predictor:

1. Latent diffusion (Rombach et al. 2022): diffuse in the VAE's 32x32x4 space instead of
   256x256x3 pixels, ~64x fewer elements, tractable on a single GPU.
2. EDM preconditioning + loss weighting (Karras et al. 2022): the achievable loss varies by
   orders of magnitude across sigma under plain VP, so easy steps drown out hard mid-range
   ones. EDM's c_skip/c_out/c_in/c_noise + lambda(sigma) keep the per-sigma loss flat by
   construction.
3. EMA of weights (decay 0.9999, EDM paper's protocol) -- raw weights are visibly noisier
   at this scale; nothing here should be evaluated without the EMA copy.

Architecture is diffusers' UNet2DModel, sized like LDM/ADM configs for 256px face datasets.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers import UNet2DModel

# works around a cuDNN SDPA bug on H100s (pytorch/pytorch#140930); falls back to flash/mem-efficient SDPA
torch.backends.cuda.enable_cudnn_sdp(False)

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "latents_path": os.environ.get("LATENTS_PATH", "checkpoints/celeba_hq_latents.pt"),
    "seed": 0,
    "batch": int(os.environ.get("BATCH", 64)),
    "lr": float(os.environ.get("LR", 1e-4)),
    "ema_decay": float(os.environ.get("EMA_DECAY", 0.9999)),
    "grad_clip": float(os.environ.get("GRAD_CLIP", 1.0)),
    "iters": int(os.environ.get("ITERS", 200_000)),
    "p_mean": float(os.environ.get("P_MEAN", -1.2)),   # EDM defaults (Table 1, ImageNet-scale)
    "p_std": float(os.environ.get("P_STD", 1.2)),
    "log_every": int(os.environ.get("LOG_EVERY", 100)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 2000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 2000)),
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/celeba_hq_edm.pt")
LOG_PATH = os.environ.get("LOG_PATH", "checkpoints/train_log.txt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/loss_curve.png")

torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)

blob = torch.load(config["latents_path"], map_location="cpu")
latents = blob["latents"].float().to(config["device"])
sigma_data = blob["sigma_data"]
config["sigma_data"] = sigma_data
n_latents, n_ch, res, _ = latents.shape
print(f"loaded {n_latents} latents, shape ({n_ch},{res},{res}), sigma_data={sigma_data:.4f}", flush=True)

net = UNet2DModel(
    sample_size=res,
    in_channels=n_ch,
    out_channels=n_ch,
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 384),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to(config["device"])
net_ema = UNet2DModel.from_config(net.config).to(config["device"])
net_ema.load_state_dict(net.state_dict())
for p in net_ema.parameters():
    p.requires_grad_(False)

opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
amp_dtype = torch.bfloat16 if config["device"] == "cuda" else torch.float32


def edm_precond(net_, x, sigma):
    # Karras et al. 2022, Table 1 ("Ours") preconditioning.
    sigma = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_noise = 0.25 * sigma.log().flatten()
    F_x = net_(c_in * x, c_noise).sample
    return c_skip * x + c_out * F_x


def edm_loss(net_, x0):
    B = x0.shape[0]
    ln_sigma = config["p_mean"] + config["p_std"] * torch.randn(B, device=x0.device)
    sigma = ln_sigma.exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    noise = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
    D_x = edm_precond(net_, x0 + noise, sigma)
    return (weight.view(-1, 1, 1, 1) * (D_x - x0) ** 2).mean()


start_iter = 0
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=config["device"])
    net.load_state_dict(ckpt["model"])
    net_ema.load_state_dict(ckpt["model_ema"])
    opt.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    print(f"resumed from {CKPT_PATH} at iter {start_iter}", flush=True)

rng = np.random.default_rng(config["seed"])
losses = []
t0 = time.time()
for it in range(start_iter, config["iters"]):
    idx = torch.from_numpy(rng.choice(n_latents, size=config["batch"], replace=True))
    x0 = latents[idx]

    with torch.autocast(device_type="cuda" if config["device"] == "cuda" else "cpu", dtype=amp_dtype,
                         enabled=config["device"] == "cuda"):
        loss = edm_loss(net, x0)

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"])
    opt.step()

    with torch.no_grad():
        d = config["ema_decay"]
        for p_ema, p in zip(net_ema.parameters(), net.parameters()):
            p_ema.mul_(d).add_(p, alpha=1 - d)

    losses.append(loss.item())

    if it % config["log_every"] == 0:
        msg = f"iter {it}  loss {loss.item():.4f}  ({time.time() - t0:.1f}s elapsed)"
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    if it % config["ckpt_every"] == 0 or it == config["iters"] - 1:
        torch.save({
            "model": net.state_dict(), "model_ema": net_ema.state_dict(),
            "opt": opt.state_dict(), "iter": it, "config": config,
        }, CKPT_PATH)

    if it % config["plot_every"] == 0 or it == config["iters"] - 1:
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.xlabel("iter"); plt.ylabel("EDM-weighted MSE"); plt.yscale("log")
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        plt.close()

print(f"done -- {CKPT_PATH} at iter {config['iters'] - 1}", flush=True)
