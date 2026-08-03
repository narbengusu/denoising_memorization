"""Fisher-Rao guided sampling (guidance.driver + guidance.basic_fr) on top of
the finetuned CIFAR-10 DDPM, mirroring qm9's sample_qm9_edm_fr.py role. Pack/
unpack are trivial here (a CIFAR-10 image IS already a flat-reshapable [3,32,32]
tensor, no padding/masking like qm9's variable-atom-count molecules)."""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

from cifar10_data import load_cifar10
from guidance.driver import guided_reverse_loop, make_autograd_guidance_fn
from guidance.basic_fr import build_flat_index, make_knn_fr_energy_fn

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
    "seed": 0,
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/ddpm_finetuned.pt")
TRAIN_IDX_PATH = os.environ.get("TRAIN_IDX_PATH", "checkpoints/train_idx.npy")
OUT_DIR = os.environ.get("OUT_DIR", "checkpoints")
N_STEPS_EVAL = int(os.environ.get("N_STEPS_EVAL", 1000))
N_SAMPLES_PER_LAMBDA = int(os.environ.get("N_SAMPLES_PER_LAMBDA", 500))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 1))
# kept as raw strings (not floats) for LAMBDA/GRAD_CLIP/WINDOW so the filename tag below is
# built from the exact token the .sh script's bash loop also sees -- reformatting through
# Python's `:g` (e.g. 1.0 -> "1") would silently diverge from bash's literal "1.0", breaking
# the memorization sweep script's filename reconstruction. Parse to float only at use site.
LAMBDA_LIST = [s.strip() for s in os.environ.get("LAMBDA_LIST", "1,5,10,50,100").split(",")]

# Same guidance hyperparameters as qm9's sample_qm9_edm_fr.py -- see its comments for the
# t_star_frac/t_end_frac/ell window semantics and why beta_t is schedule-derived, not free.
# GRAD_CLIP_LIST and WINDOW_LIST make this a genuine lambda x grad_clip x window outer-grid
# sweep, matching experiments/so_n/03_fr_guidance.ipynb's wider search -- lambda's right
# scale is entangled with grad_clip (effective step ~ lam * a(t) * min(||grad||, grad_clip)),
# so sweeping lambda alone can hide the actual optimum. Values here are NOT copied from the
# SO(n) notebook's grid -- CIFAR-10 pixels (range [-1,1]) have a different natural
# gradient-norm scale than SO(n)'s toy rotation matrices, so each needs its own calibration.
K_NN = int(os.environ.get("K_NN", 16))
GRAD_CLIP_LIST = [s.strip() for s in os.environ.get("GRAD_CLIP_LIST", "1.0").split(",")]
WINDOW_LIST = [s.strip() for s in os.environ.get("WINDOW_LIST", "0.6-1.0").split(",")]
ELL = int(os.environ.get("ELL", 20))
IOTA_MIN = os.environ.get("IOTA_MIN")
IOTA_MAX = os.environ.get("IOTA_MAX")
TARGET_RANGE = (float(IOTA_MIN), float(IOTA_MAX)) if IOTA_MIN is not None and IOTA_MAX is not None else None

# Diagnostic-only: full per-sample I(y) distribution at these VP-SDE-style t fractions
# (1.0 = pure noise / first reverse step, 0.0 = clean data / last step), independent of
# the guidance window/period gate above -- lets you see e.g. whether I collapses toward
# memorized training points only late in sampling (small t) or throughout.
T_HIST_FRACS = [float(s.strip()) for s in os.environ.get("T_HIST_FRACS", "0.2,0.4,0.6,0.8,1.0").split(",")]

torch.manual_seed(config["seed"])
os.makedirs(OUT_DIR, exist_ok=True)

D_FLAT = 3 * 32 * 32


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 3, 32, 32)


images, _ = load_cifar10(split="train")
train_idx = np.load(TRAIN_IDX_PATH)
X_train = images[train_idx].to(config["device"])
train_index = build_flat_index(pack(X_train))
print(f"built flat training index: {train_index.shape}", flush=True)

scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
scheduler.set_timesteps(N_STEPS_EVAL)
alphas_cumprod = scheduler.alphas_cumprod.to(config["device"])

# scheduler.timesteps descends from ~num_train_timesteps-1 (pure noise) to 0 (clean data),
# matching this project's t in [0, 1] convention (t=1 noisiest) once normalized -- map each
# requested T_HIST_FRACS entry to the nearest actual step_idx (1-indexed, same axis
# guided_reverse_loop iterates on) once, up front, since it's the same for every combo below.
num_train_timesteps = scheduler.config.num_train_timesteps
t_norm = scheduler.timesteps.float() / (num_train_timesteps - 1)
hist_step_idx = {}
for t_frac in T_HIST_FRACS:
    step_idx = int(torch.argmin((t_norm - t_frac).abs()).item()) + 1
    hist_step_idx[t_frac] = step_idx
print(f"I-distribution histogram points: "
      + ", ".join(f"t={t_frac} -> step {s}" for t_frac, s in hist_step_idx.items()), flush=True)
MEASURE_STEPS = set(hist_step_idx.values())

net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
ckpt = torch.load(CKPT_PATH, map_location=config["device"])
net.load_state_dict(ckpt.get("model_ema", ckpt["model"]))
net.eval()
print(f"loaded checkpoint from {CKPT_PATH} (using "
      f"{'EMA' if 'model_ema' in ckpt else 'raw'} weights, trained {ckpt['iter'] + 1} iterations)", flush=True)


@torch.no_grad()
def fr_guided_sample(B, n_steps, lam, grad_clip, t_star_frac, t_end_frac, generator):
    timesteps = scheduler.timesteps  # length n_steps, descending
    z = pack(torch.randn(B, 3, 32, 32, device=config["device"], generator=generator))

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

    def step_scale_fn(step_idx):
        t = timesteps[step_idx - 1]
        # beta_tilde_t = (1-abar_{t-1})/(1-abar_t) * beta_t, Ho et al.'s POSTERIOR
        # variance of q(x_{t-1}|x_t,x_0) -- same role as qm9's post_sigma^2 (see its
        # sample_qm9_edm_fr.py step_scale_fn comment). variance_type="fixed_small" is
        # forced explicitly: google/ddpm-cifar10-32's scheduler_config.json declares
        # variance_type="fixed_large", under which scheduler._get_variance(t) (no
        # override) silently returns the raw forward beta_t instead -- strictly
        # larger than beta_tilde_t and NOT what step_scale_fn is supposed to return.
        return scheduler._get_variance(t, variance_type="fixed_small")

    def base_step_with_t(z_t, step_idx):
        y_fn.t = timesteps[step_idx - 1]
        return base_step(z_t, step_idx)

    def beta_t_fn():
        return 1 - alphas_cumprod[y_fn.t]

    energy_fn = make_knn_fr_energy_fn(train_index, K_NN, lambda: beta_t_fn().expand(B))
    guidance_fn = make_autograd_guidance_fn(y_fn, energy_fn, target_range=TARGET_RANGE, grad_clip=grad_clip)

    def measure_fn(z_t, step_idx):
        y_fn.t = timesteps[step_idx - 1]  # beta_t_fn() below reads y_fn.t, same as base_step_with_t sets it
        with torch.no_grad():
            return energy_fn(y_fn(z_t))

    I_hist = {}
    z_final, I_log = guided_reverse_loop(
        z, n_steps, base_step_with_t, guidance_fn,
        progress_lo=t_star_frac, progress_hi=t_end_frac, ell=ELL, lam=lam, step_scale_fn=step_scale_fn,
        measure_fn=measure_fn, measure_steps=MEASURE_STEPS, hist_out=I_hist,
    )
    I_hist_by_tfrac = {t_frac: I_hist[step_idx] for t_frac, step_idx in hist_step_idx.items()}
    return unpack(z_final).clamp(-1, 1).cpu(), I_log, I_hist_by_tfrac


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
            x_gen, I_log, I_hist_by_tfrac = fr_guided_sample(
                N_SAMPLES_PER_LAMBDA, N_STEPS_EVAL, lam, grad_clip, t_star_frac, t_end_frac, gen)
            dt_sample = time.time() - t0
            tag = f"lambda{lam_str}_clip{clip_str}_win{window_str}"
            print(f"{tag}: generated {N_SAMPLES_PER_LAMBDA} images in {dt_sample:.1f}s  I_beta trace: "
                  + ", ".join(f"(step {s}, I={i:.4f}, n_active={n})" for s, i, n in I_log), flush=True)
            print(f"{tag}: I distribution -- "
                  + ", ".join(f"(t={t_frac}, mean={I_t.mean():.4f}, std={I_t.std():.4f})"
                              for t_frac, I_t in I_hist_by_tfrac.items()), flush=True)

            out_path = os.path.join(OUT_DIR, f"samples_fr_{tag}.pt")
            torch.save({
                "x_gen": x_gen, "n_eval": N_SAMPLES_PER_LAMBDA, "n_steps_eval": N_STEPS_EVAL,
                "ckpt_iter": int(ckpt["iter"]), "ckpt_path": CKPT_PATH,
                "lambda_guide": lam, "beta_fr": "schedule (sigma_t^2)", "k_nn": K_NN,
                "grad_clip": grad_clip, "t_star_frac": t_star_frac, "t_end_frac": t_end_frac,
                "ell": ELL, "target_range": TARGET_RANGE, "I_log": I_log,
                "I_hist_by_tfrac": I_hist_by_tfrac, "t_hist_fracs": T_HIST_FRACS,
            }, out_path)
            print(f"saved {tag} samples to {out_path}", flush=True)

            fig, axes = plt.subplots(1, len(T_HIST_FRACS), figsize=(4 * len(T_HIST_FRACS), 3.2), sharey=True)
            if len(T_HIST_FRACS) == 1:
                axes = [axes]
            for ax, t_frac in zip(axes, T_HIST_FRACS):
                ax.hist(I_hist_by_tfrac[t_frac].numpy(), bins=30, color="steelblue")
                ax.set(xlabel="I(y)", title=f"t={t_frac}")
            axes[0].set_ylabel("count")
            fig.suptitle(f"I(y) distribution over batch -- {tag}")
            plt.tight_layout()
            i_hist_path = os.path.join(OUT_DIR, f"i_hist_{tag}.png")
            plt.savefig(i_hist_path, dpi=120)
            plt.close(fig)
            print(f"saved I-distribution histogram to {i_hist_path}", flush=True)
