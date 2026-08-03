"""Empirical score-divergence guided sampling (guidance.score_divergence) on the
finetuned CIFAR-10 DDPM -- no training-atom/image index needed, unlike
sample_cifar10_fr.py. Mirrors qm9's sample_qm9_edm_scorevar.py; GUIDANCE_GRAD_MODE=fd
(default) avoids double-backward through the whole UNet, same OOM concern noted
there for the EGNN."""
import os
import time
import torch
from diffusers import UNet2DModel, DDPMScheduler

from guidance.driver import guided_reverse_loop, make_autograd_guidance_fn
from guidance.score_divergence import make_score_div_energy_fn, make_score_div_fd_guidance_fn

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
    "seed": 0,
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/ddpm_finetuned.pt")
OUT_DIR = os.environ.get("OUT_DIR", "checkpoints")
N_STEPS_EVAL = int(os.environ.get("N_STEPS_EVAL", 1000))
N_SAMPLES_PER_LAMBDA = int(os.environ.get("N_SAMPLES_PER_LAMBDA", 500))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 1))
# kept as raw strings (not floats) so the filename tag is built from the exact token the
# .sh script's bash loop also sees -- see the identical comment in sample_cifar10_fr.py.
LAMBDA_LIST = [s.strip() for s in os.environ.get("LAMBDA_LIST", "1,5,10,50,100").split(",")]

# GRAD_CLIP_LIST and WINDOW_LIST make this a genuine lambda x grad_clip x window outer-grid
# sweep -- see the identical comment in sample_cifar10_fr.py for why (and why the SO(n)
# notebook's grid values don't transfer directly to CIFAR-10's pixel scale).
R_PROBES = int(os.environ.get("R_PROBES", 4))
WINDOW_LIST = [s.strip() for s in os.environ.get("WINDOW_LIST", "0.6-1.0").split(",")]
ELL = int(os.environ.get("ELL", 20))
GRAD_CLIP_LIST = [s.strip() for s in os.environ.get("GRAD_CLIP_LIST", "1.0").split(",")]
IOTA_MIN = os.environ.get("IOTA_MIN")
IOTA_MAX = os.environ.get("IOTA_MAX")
TARGET_RANGE = (float(IOTA_MIN), float(IOTA_MAX)) if IOTA_MIN is not None and IOTA_MAX is not None else None

GUIDANCE_GRAD_MODE = os.environ.get("GUIDANCE_GRAD_MODE", "fd")
N_DIRS = int(os.environ.get("N_DIRS", 4))
FD_EPS = float(os.environ.get("FD_EPS", 1e-2))

torch.manual_seed(config["seed"])
os.makedirs(OUT_DIR, exist_ok=True)

D_FLAT = 3 * 32 * 32


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 3, 32, 32)


scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
scheduler.set_timesteps(N_STEPS_EVAL)
alphas_cumprod = scheduler.alphas_cumprod.to(config["device"])

net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
ckpt = torch.load(CKPT_PATH, map_location=config["device"])
net.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
net.eval()
print(f"loaded checkpoint from {CKPT_PATH} (using "
      f"{'EMA' if 'model_ema' in ckpt else 'raw'} weights, trained {ckpt['iter'] + 1} iterations)", flush=True)


@torch.no_grad()
def scorevar_guided_sample(B, n_steps, lam, grad_clip, t_star_frac, t_end_frac, generator):
    timesteps = scheduler.timesteps
    z = pack(torch.randn(B, 3, 32, 32, device=config["device"], generator=generator))
    d_real = torch.full((B,), float(D_FLAT), device=config["device"])

    def denoise_to_y(z_t, t):
        x_t = unpack(z_t)
        eps_hat = net(x_t, t).sample
        a_t = alphas_cumprod[t]
        y = (x_t - (1 - a_t).sqrt() * eps_hat) / a_t.sqrt()
        return pack(y).clamp(-1, 1)

    def base_step(z_t, step_idx):
        t = timesteps[step_idx - 1]
        x_t = unpack(z_t)
        eps_hat = net(x_t, t).sample
        x_next = scheduler.step(eps_hat, t, x_t, generator=generator).prev_sample
        return pack(x_next)

    def y_fn(z_t):
        return denoise_to_y(z_t, y_fn.t)

    def score_fn(y):
        # s_theta(y, t) = -eps_theta(y, t) / sigma_t, at the SAME t as y_fn (see qm9's
        # sample_qm9_edm_scorevar.py score_fn comment for the reasoning).
        x_y = unpack(y)
        t = y_fn.t
        eps_hat = net(x_y, t).sample
        sigma_t = (1 - alphas_cumprod[t]).sqrt()
        return pack(-eps_hat / sigma_t)

    def sigma2_fn():
        return (1 - alphas_cumprod[y_fn.t]).expand(B)

    def d_fn():
        return d_real

    def step_scale_fn(step_idx):
        t = timesteps[step_idx - 1]
        # beta_tilde_t, forced via variance_type="fixed_small" -- see the identical
        # comment in sample_cifar10_fr.py's step_scale_fn for why the override is
        # required (google/ddpm-cifar10-32's scheduler_config.json declares
        # variance_type="fixed_large", which would silently return raw beta_t instead).
        return scheduler._get_variance(t, variance_type="fixed_small")

    def base_step_with_t(z_t, step_idx):
        y_fn.t = timesteps[step_idx - 1]
        return base_step(z_t, step_idx)

    if GUIDANCE_GRAD_MODE == "fd":
        guidance_fn = make_score_div_fd_guidance_fn(
            y_fn, score_fn, sigma2_fn, d_fn, R_PROBES,
            n_dirs=N_DIRS, eps=FD_EPS, target_range=TARGET_RANGE, grad_clip=grad_clip,
        )
    elif GUIDANCE_GRAD_MODE == "autograd":
        energy_fn = make_score_div_energy_fn(score_fn, sigma2_fn, d_fn, R_PROBES)
        guidance_fn = make_autograd_guidance_fn(y_fn, energy_fn, target_range=TARGET_RANGE, grad_clip=grad_clip)
    else:
        raise ValueError(f"unknown GUIDANCE_GRAD_MODE={GUIDANCE_GRAD_MODE!r}, expected 'fd' or 'autograd'")

    z_final, I_log = guided_reverse_loop(
        z, n_steps, base_step_with_t, guidance_fn,
        progress_lo=t_star_frac, progress_hi=t_end_frac, ell=ELL, lam=lam, step_scale_fn=step_scale_fn,
    )
    return unpack(z_final).clamp(-1, 1).cpu(), I_log


n_combos = len(LAMBDA_LIST) * len(GRAD_CLIP_LIST) * len(WINDOW_LIST)
print(f"sweeping {len(LAMBDA_LIST)} lambda x {len(GRAD_CLIP_LIST)} grad_clip x "
      f"{len(WINDOW_LIST)} window = {n_combos} combos", flush=True)

for window_str in WINDOW_LIST:
    t_star_frac, t_end_frac = (float(v) for v in window_str.split("-"))
    for clip_str in GRAD_CLIP_LIST:
        grad_clip = float(clip_str)
        for lam_str in LAMBDA_LIST:
            lam = float(lam_str)
            gen = torch.Generator(device=config["device"]).manual_seed(SAMPLE_SEED)
            t0 = time.time()
            x_gen, I_log = scorevar_guided_sample(
                N_SAMPLES_PER_LAMBDA, N_STEPS_EVAL, lam, grad_clip, t_star_frac, t_end_frac, gen)
            dt_sample = time.time() - t0
            tag = f"lambda{lam_str}_clip{clip_str}_win{window_str}"
            print(f"{tag}: generated {N_SAMPLES_PER_LAMBDA} images in {dt_sample:.1f}s  I_beta trace: "
                  + ", ".join(f"(step {s}, I={i:.4f}, n_active={n})" for s, i, n in I_log), flush=True)

            out_path = os.path.join(OUT_DIR, f"samples_scorevar_{tag}.pt")
            torch.save({
                "x_gen": x_gen, "n_eval": N_SAMPLES_PER_LAMBDA, "n_steps_eval": N_STEPS_EVAL,
                "ckpt_iter": int(ckpt["iter"]), "ckpt_path": CKPT_PATH,
                "lambda_guide": lam, "r_probes": R_PROBES, "guidance_grad_mode": GUIDANCE_GRAD_MODE,
                "grad_clip": grad_clip, "t_star_frac": t_star_frac, "t_end_frac": t_end_frac,
                "ell": ELL, "target_range": TARGET_RANGE, "I_log": I_log,
            }, out_path)
            print(f"saved {tag} samples to {out_path}", flush=True)
