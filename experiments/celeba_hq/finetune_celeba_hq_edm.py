"""Finetunes the already-trained (full-30k, ~0% memorized) celeba_hq_edm
checkpoint on a SMALL subset of the precomputed latents (N_TRAIN images,
default 2000) to induce heavy memorization -- same recipe as cifar10's
finetune_cifar10_ddpm.py, grounded in the memorization literature (Carlini
et al. 2023; Gu et al. "On Memorization in Diffusion Models"): a model
trained on a large enough dataset generalizes, but continuing training on a
small fixed subset until convergence reliably memorizes it. Starting from
the fully-trained checkpoint (rather than from scratch) means the finetune
only has to specialize onto the small subset, not learn faces from zero.

Same subset (same seed) is saved to checkpoints/celeba_hq_finetune_train_idx.npy
so 01_sample_celeba_hq.ipynb can be pointed at exactly what this run
could have possibly memorized, the same way cifar10's assess script uses
checkpoints/train_idx.npy. Indices are into the precomputed 60k-latent pool
(30k images x 2 for the horizontal-flip augmentation), not the raw image
files directly.
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

torch.backends.cuda.enable_cudnn_sdp(False)  # see train_celeba_hq_edm.py's docstring

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "latents_path": os.environ.get("LATENTS_PATH", "checkpoints/celeba_hq_latents.pt"),
    "base_ckpt_path": os.environ.get("BASE_CKPT_PATH", "checkpoints/celeba_hq_edm.pt"),
    "seed": 0,
    "n_train": int(os.environ.get("N_TRAIN", 200)),   # small on purpose -- induces memorization
    "batch": int(os.environ.get("BATCH", 64)),
    "lr": float(os.environ.get("LR", 2e-5)),            # lower than the base run's 1e-4 -- finetuning, not training from scratch
    "ema_decay": float(os.environ.get("EMA_DECAY", 0.999)),
    "grad_clip": float(os.environ.get("GRAD_CLIP", 1.0)),
    "iters": int(os.environ.get("ITERS", 20_000)),
    "p_mean": float(os.environ.get("P_MEAN", -1.2)),
    "p_std": float(os.environ.get("P_STD", 1.2)),
    "log_every": int(os.environ.get("LOG_EVERY", 100)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 1000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 1000)),
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/celeba_hq_edm_finetuned.pt")
TRAIN_IDX_PATH = os.environ.get("TRAIN_IDX_PATH", "checkpoints/celeba_hq_finetune_train_idx.npy")
LOG_PATH = os.environ.get("LOG_PATH", "checkpoints/finetune_train_log.txt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/finetune_loss_curve.png")

torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)

blob = torch.load(config["latents_path"], map_location="cpu")
all_latents = blob["latents"].float()
sigma_data = blob["sigma_data"]
config["sigma_data"] = sigma_data
n_ch, res = all_latents.shape[1], all_latents.shape[2]

rng = np.random.default_rng(config["seed"])
train_idx = rng.choice(all_latents.shape[0], size=config["n_train"], replace=False)
np.save(TRAIN_IDX_PATH, train_idx)
latents = all_latents[train_idx].to(config["device"])
print(f"finetuning on {config['n_train']} latents (seed={config['seed']}), "
      f"shape ({n_ch},{res},{res}), sigma_data={sigma_data:.4f}", flush=True)

net = UNet2DModel(
    sample_size=res, in_channels=n_ch, out_channels=n_ch, layers_per_block=2,
    block_out_channels=(128, 256, 384, 384),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to(config["device"])
net_ema = UNet2DModel.from_config(net.config).to(config["device"])

base_ckpt = torch.load(config["base_ckpt_path"], map_location=config["device"])
net.load_state_dict(base_ckpt["model_ema"])       # start from the trained EMA weights, not raw
net_ema.load_state_dict(base_ckpt["model_ema"])
for p in net_ema.parameters():
    p.requires_grad_(False)
print(f"initialized from {config['base_ckpt_path']} (iter {base_ckpt['iter']})", flush=True)

opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
amp_dtype = torch.bfloat16 if config["device"] == "cuda" else torch.float32


def edm_precond(net_, x, sigma):
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
    print(f"resumed finetune from {CKPT_PATH} at iter {start_iter}", flush=True)

losses = []
t0 = time.time()
for it in range(start_iter, config["iters"]):
    idx = torch.from_numpy(rng.choice(config["n_train"], size=config["batch"], replace=True))
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
