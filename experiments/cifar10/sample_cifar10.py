"""Generates N_EVAL images from the finetuned checkpoint using diffusers'
own ancestral DDPM reverse loop (scheduler.step), saves everything to
checkpoints/samples.pt."""
import os
import time
import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/ddpm_finetuned.pt")
OUT_PATH = os.environ.get("OUT_PATH", "checkpoints/samples.pt")
GRID_PATH = os.environ.get("GRID_PATH", "checkpoints/samples_grid.png")
N_EVAL = int(os.environ.get("N_EVAL", 500))
N_STEPS_EVAL = int(os.environ.get("N_STEPS_EVAL", 1000))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 1))

os.makedirs("checkpoints", exist_ok=True)

scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
scheduler.set_timesteps(N_STEPS_EVAL)

net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
ckpt = torch.load(CKPT_PATH, map_location=config["device"])
net.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
net.eval()
print(f"loaded checkpoint from {CKPT_PATH} (using "
      f"{'EMA' if 'model_ema' in ckpt else 'raw'} weights, trained {ckpt['iter'] + 1} iterations)", flush=True)


@torch.no_grad()
def sample(net, B, generator):
    x = torch.randn(B, 3, 32, 32, device=config["device"], generator=generator)
    for t in scheduler.timesteps:
        eps_hat = net(x, t).sample
        x = scheduler.step(eps_hat, t, x, generator=generator).prev_sample
    return x.clamp(-1, 1)


t0 = time.time()
gen = torch.Generator(device=config["device"]).manual_seed(SAMPLE_SEED)
x_gen = sample(net, N_EVAL, gen)
print(f"generated {N_EVAL} images with n_steps={N_STEPS_EVAL} in {time.time() - t0:.1f}s", flush=True)

save_image((x_gen[:64] + 1) / 2, GRID_PATH, nrow=8)

torch.save({
    "x_gen": x_gen.cpu(), "n_eval": N_EVAL, "n_steps_eval": N_STEPS_EVAL,
    "ckpt_iter": int(ckpt["iter"]), "ckpt_path": CKPT_PATH, "sample_seed": SAMPLE_SEED,
}, OUT_PATH)
print(f"saved samples to {OUT_PATH}, preview grid to {GRID_PATH}", flush=True)
