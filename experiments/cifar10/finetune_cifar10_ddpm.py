"""Finetunes the pretrained google/ddpm-cifar10-32 UNet on a SMALL subset of
CIFAR-10 (N_TRAIN images, default 2000) to induce heavy memorization -- per
the memorization literature (Carlini et al. 2023; Gu et al. "On Memorization
in Diffusion Models"), the full-50k pretrained checkpoint itself is not
memorized, but continuing training on a small fixed subset until convergence
reliably is. Same subset (same seed) is reused by sample_cifar10.py and
assess_memorization_cifar10.py so "nearest training image" always means
"nearest image this run could have possibly memorized."
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

from cifar10_data import load_cifar10

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
    "seed": 0,
    "n_train": int(os.environ.get("N_TRAIN", 2000)),   # small on purpose -- induces memorization
    "batch": int(os.environ.get("BATCH", 64)),
    "lr": float(os.environ.get("LR", 2e-5)),
    "ema_decay": float(os.environ.get("EMA_DECAY", 0.999)),
    "iters": int(os.environ.get("ITERS", 20_000)),
    "log_every": int(os.environ.get("LOG_EVERY", 100)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 1000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 1000)),
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/ddpm_finetuned.pt")
LOG_PATH = os.environ.get("LOG_PATH", "checkpoints/train_log.txt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/loss_curve.png")

torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)

images, _ = load_cifar10(split="train")
rng = np.random.default_rng(config["seed"])
train_idx = rng.choice(len(images), size=config["n_train"], replace=False)
np.save("checkpoints/train_idx.npy", train_idx)
X_train = images[train_idx].to(config["device"])
print(f"finetuning on {config['n_train']} CIFAR-10 images (seed={config['seed']})", flush=True)

net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
net_ema = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
net_ema.load_state_dict(net.state_dict())
for p in net_ema.parameters():
    p.requires_grad_(False)

opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
n_steps = scheduler.config.num_train_timesteps

start_iter = 0
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=config["device"])
    net.load_state_dict(ckpt["model"])
    net_ema.load_state_dict(ckpt["model_ema"])
    opt.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    print(f"resumed from {CKPT_PATH} at iter {start_iter}", flush=True)

losses = []
t0 = time.time()
for it in range(start_iter, config["iters"]):
    idx = torch.from_numpy(rng.choice(config["n_train"], size=config["batch"], replace=True))
    x0 = X_train[idx]
    noise = torch.randn_like(x0)
    t = torch.randint(0, n_steps, (config["batch"],), device=config["device"]).long()
    x_t = scheduler.add_noise(x0, noise, t)

    eps_hat = net(x_t, t).sample
    loss = F.mse_loss(eps_hat, noise)

    opt.zero_grad()
    loss.backward()
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
            "pretrained_dir": config["pretrained_dir"],
        }, CKPT_PATH)

    if it % config["plot_every"] == 0 or it == config["iters"] - 1:
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.xlabel("iter"); plt.ylabel("noise-prediction MSE"); plt.yscale("log")
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        plt.close()

print(f"done -- checkpoints/ddpm_finetuned.pt at iter {config['iters'] - 1}", flush=True)
