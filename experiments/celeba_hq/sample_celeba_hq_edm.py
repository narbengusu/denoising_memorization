"""Samples from the EMA weights of a trained celeba_hq_edm checkpoint using EDM's
deterministic 2nd-order Heun sampler (Karras et al. 2022, Algorithm 2) -- fewer steps to a
given quality than plain ancestral Euler-Maruyama, since the 2nd-order correction cancels
most of the local discretization error. Always samples from net_ema, never raw weights.

Decodes latents back to pixels with the same frozen VAE used to encode them.
"""
import os
import torch
import torchvision.utils as vutils
from diffusers import UNet2DModel, AutoencoderKL

# see train_celeba_hq_edm.py's docstring -- same H100 cuDNN SDPA bug hits the UNet's
# attention blocks here too.
torch.backends.cuda.enable_cudnn_sdp(False)

device = os.environ.get("DEVICE", "cpu")
ckpt_path = os.environ.get("CKPT_PATH", "checkpoints/celeba_hq_edm.pt")
vae_dir = os.environ.get("VAE_DIR", "pretrained/sd-vae-ft-mse")
out_path = os.environ.get("OUT_PATH", "checkpoints/samples.pt")
grid_path = os.environ.get("GRID_PATH", "checkpoints/samples_grid.png")
n_eval = int(os.environ.get("N_EVAL", 64))
n_steps = int(os.environ.get("N_STEPS_EVAL", 32))
sigma_min = float(os.environ.get("SIGMA_MIN", 0.002))
sigma_max = float(os.environ.get("SIGMA_MAX", 80.0))
rho = float(os.environ.get("RHO", 7.0))
seed = int(os.environ.get("SAMPLE_SEED", 1))

ckpt = torch.load(ckpt_path, map_location=device)
sigma_data = ckpt["config"]["sigma_data"]

# reconstruct architecture exactly as trained (must match train_celeba_hq_edm.py)
n_ch, res = 4, 32
net = UNet2DModel(
    sample_size=res, in_channels=n_ch, out_channels=n_ch, layers_per_block=2,
    block_out_channels=(128, 256, 384, 384),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to(device)
net.load_state_dict(ckpt["model_ema"])
net.eval()


def edm_precond(x, sigma):
    sigma = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_noise = 0.25 * sigma.log().flatten()
    F_x = net(c_in * x, c_noise).sample
    return c_skip * x + c_out * F_x


torch.manual_seed(seed)
step_idx = torch.arange(n_steps, dtype=torch.float64, device=device)
sigmas = (sigma_max ** (1 / rho) + step_idx / (n_steps - 1) *
          (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
sigmas = torch.cat([sigmas, torch.zeros(1, device=device)]).float()

x = torch.randn(n_eval, n_ch, res, res, device=device) * sigma_max
with torch.no_grad():
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        d_cur = (x - edm_precond(x, s_cur.expand(n_eval))) / s_cur
        x_next = x + (s_next - s_cur) * d_cur
        if s_next > 0:
            d_next = (x_next - edm_precond(x_next, s_next.expand(n_eval))) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        x = x_next

vae = AutoencoderKL.from_pretrained(vae_dir).to(device).eval()
with torch.no_grad():
    images = vae.decode(x / vae.config.scaling_factor).sample
images = images.clamp(-1, 1)

torch.save(images.cpu(), out_path)
vutils.save_image(images, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
print(f"sampled {n_eval} images -- saved {out_path} and {grid_path}", flush=True)
