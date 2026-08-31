"""FR-guidance pipeline for the finetuned CIFAR-10 DDPM -- same narrative as
experiments/toy_manifold/02_fr_guidance.ipynb / experiments/so_n/03_fr_guidance.ipynb (the
k-NN Fisher-Rao energy, unprojected + tangent-projected, a gradient-norm diagnostic, a full
jacobian x r' x window x grad_clip x lam sweep, per-parameter marginals), adapted to
a headless script (prints + savefig, no inline display) and to this project's low-rank JacNet
(train_cifar10_jacnet.py) rather than a dense one.

jacobian="jacnet" is swept over R_PRIME_LIST (default 1,2,3,5) -- how many of JacNet's
identified normal (high-curvature) directions to project OUT via
driver.make_lowrank_normal_projector_fn, the inverse framing from the toy/SO(3) notebooks'
proj_k (see train_cifar10_jacnet.py's module docstring for why: with a low-rank M, "keep the
smallest-k-eigenvalue directions" would be numerically arbitrary).

Metrics are cheap approximations for the SWEEP (mem_ratio via nearest-training-image distance,
fidelity_proxy via nearest-HELD-OUT-image distance, both reusing
assess_memorization_cifar10.py's nearest_neighbor_d1_d2/classify_images) -- NOT full FID, which
would be far too slow to compute per grid cell. Run assess_memorization_cifar10.py separately
(real FID) on whichever specific combo's samples_*.pt looks best afterward.

SCALE WARNING: sample_cifar10_fr.py's existing single-method grid (30 combos, 500
samples/1000 steps) already needs ~8h wall time. This script's default grid is deliberately
much smaller (2 windows x 3 grad_clips x 3 lams, 100 samples/250 steps) since it also sweeps
method x jacobian x r' on top -- widen via the env vars below once you've narrowed things down.

GRAD_CLIP_LIST entries are either a plain number (fixed cap, e.g. "10") or "adaptive:<scale>"
(e.g. "adaptive:1.0"), mixed freely in the same list -- see resolve_clip_spec /
make_adaptive_clip_fn below. Adaptive caps the guidance correction at scale * ||s_theta(z_t,
t)||, the model's own ambient score norm at the CURRENT (noisy) z_t, not the denoised y --
driver.py's _resolve_clip_norm docstring's proposed use case for its callable-grad_clip path,
here actually wired up for the first time (previously only a mechanism, unused by any script
or notebook).
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

from cifar10_data import load_cifar10
from assess_memorization_cifar10 import nearest_neighbor_d1_d2, classify_images
from guidance import driver, basic_fr

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
    "seed": 0,
    "n_samples": int(os.environ.get("N_SAMPLES", 100)),
    "n_steps_eval": int(os.environ.get("N_STEPS_EVAL", 250)),
    "k_neighbors": int(os.environ.get("K_NN", 5)),
    "ell": int(os.environ.get("ELL", 1)),
    "mem_ratio_threshold": float(os.environ.get("MEM_RATIO_THRESHOLD", 1 / 3)),
    "n_heldout": int(os.environ.get("N_HELDOUT", 500)),
}
SCORE_CKPT_PATH = os.environ.get("SCORE_CKPT_PATH", "checkpoints/ddpm_finetuned.pt")
JACNET_CKPT_PATH = os.environ.get("JACNET_CKPT_PATH", "checkpoints/cifar10_jacnet.pt")
TRAIN_IDX_PATH = os.environ.get("TRAIN_IDX_PATH", "checkpoints/train_idx.npy")
OUT_DIR = os.environ.get("OUT_DIR", "checkpoints")
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 1))

# kept as raw strings so the filename/row tag is built from the exact token the .sh script's
# bash loop also sees -- see the identical comment in sample_cifar10_fr.py.
LAMBDA_LIST = [s.strip() for s in os.environ.get("LAMBDA_LIST", "1,10,50").split(",")]
GRAD_CLIP_LIST = [s.strip() for s in os.environ.get("GRAD_CLIP_LIST", "10,100,adaptive:1.0").split(",")]
WINDOW_LIST = [s.strip() for s in os.environ.get("WINDOW_LIST", "0.3-1.0,0.6-1.0").split(",")]
R_PRIME_LIST = [int(s.strip()) for s in os.environ.get("R_PRIME_LIST", "1,2,3,5").split(",")]

IOTA_MIN = os.environ.get("IOTA_MIN")
IOTA_MAX = os.environ.get("IOTA_MAX")
TARGET_RANGE = (float(IOTA_MIN), float(IOTA_MAX)) if IOTA_MIN is not None and IOTA_MAX is not None else None

DIAG_N_STEPS = int(os.environ.get("DIAG_N_STEPS", 100))   # separate, cheaper schedule for the diagnostic panel
DIAG_STRIDE = int(os.environ.get("DIAG_STRIDE", 5))       # only compute the guidance grad every this-many diag steps

torch.manual_seed(config["seed"])
os.makedirs(OUT_DIR, exist_ok=True)

D_FLAT = 3 * 32 * 32


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 3, 32, 32)


# ---------------------------------------------------------------------------
# Data: training subset (memorization target) + held-out test subset (fidelity_proxy target)
# ---------------------------------------------------------------------------
images, _ = load_cifar10(split="train")
train_idx = np.load(TRAIN_IDX_PATH)
X_train = images[train_idx].to(config["device"])
train_index = basic_fr.build_flat_index(pack(X_train))
print(f"built flat training index: {train_index.shape}", flush=True)

test_images, _ = load_cifar10(split="test")
rng = np.random.default_rng(0)
heldout_idx = rng.choice(len(test_images), size=min(config["n_heldout"], len(test_images)), replace=False)
X_heldout = pack(test_images[heldout_idx]).to(config["device"])

# ---------------------------------------------------------------------------
# Frozen fine-tuned score net + scheduler
# ---------------------------------------------------------------------------
net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
score_ckpt = torch.load(SCORE_CKPT_PATH, map_location=config["device"])
net.load_state_dict(score_ckpt.get("model_ema", score_ckpt["model"]))
net.eval()
print(f"loaded fine-tuned score net from {SCORE_CKPT_PATH} (using "
      f"{'EMA' if 'model_ema' in score_ckpt else 'raw'} weights, "
      f"trained {score_ckpt['iter'] + 1} iterations)", flush=True)

scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
num_train_timesteps = scheduler.config.num_train_timesteps


# ---------------------------------------------------------------------------
# Frozen low-rank JacNet -- identical architecture to train_cifar10_jacnet.py (duplicated
# here rather than imported, matching this project's convention elsewhere of redefining
# model classes per consuming script; importing would re-run that script's training loop
# as a side effect, since it has no `if __name__ == "__main__"` guard).
# ---------------------------------------------------------------------------
class JacNet(nn.Module):
    def __init__(self, d, hidden, depth, rank):
        super().__init__()
        self.d, self.rank = d, rank
        self.input_proj = nn.Linear(d + 1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth * 2)])
        self.output_proj = nn.Linear(hidden, d * rank)

    def forward(self, z, t):
        h = F.silu(self.input_proj(torch.cat([z, t[:, None]], dim=-1)))
        for i in range(0, len(self.layers), 2):
            r = F.silu(self.layers[i](h))
            r = self.layers[i + 1](r)
            h = h + r
        return self.output_proj(h).reshape(z.shape[0], self.d, self.rank)


jacnet_ckpt = torch.load(JACNET_CKPT_PATH, map_location=config["device"])
jac_config = jacnet_ckpt["config"]
jac_net = JacNet(D_FLAT, jac_config["jac_hidden"], jac_config["jac_depth"], jac_config["jac_rank"]).to(config["device"])
jac_net.load_state_dict(jacnet_ckpt["model"])
jac_net.eval()
for p in jac_net.parameters():
    p.requires_grad_(False)
JAC_RANK = jac_config["jac_rank"]
assert max(R_PRIME_LIST) <= JAC_RANK, f"R_PRIME_LIST has a value > JAC_RANK={JAC_RANK} the JacNet checkpoint was trained with"
print(f"loaded low-rank JacNet from {JACNET_CKPT_PATH} (rank={JAC_RANK}, "
      f"trained {jacnet_ckpt['iter'] + 1} iterations)", flush=True)


# ---------------------------------------------------------------------------
# Wire up the generic interface -- same y_fn.t / base_step pattern as
# sample_cifar10_fr.py / sample_cifar10_scorevar.py.
# ---------------------------------------------------------------------------
def make_wiring(n_steps, generator=None):
    scheduler.set_timesteps(n_steps)
    timesteps = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(config["device"])

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

    def base_step_with_t(z_t, step_idx):
        y_fn.t = timesteps[step_idx - 1]
        return base_step(z_t, step_idx)

    def step_scale_fn(step_idx):
        t = timesteps[step_idx - 1]
        # beta_tilde_t via variance_type="fixed_small" -- see the identical comment in
        # sample_cifar10_fr.py's step_scale_fn for why the override is required.
        return scheduler._get_variance(t, variance_type="fixed_small")

    def score_at_zt_fn(z_t):
        # s_theta = -eps_theta/sigma_t evaluated at the current NOISY z_t rather than the
        # denoised y -- the adaptive-clip cap's job is to bound the guidance correction
        # relative to the model's own drift at the point actually being stepped, so it needs
        # the score there, not at a would-be clean estimate.
        x_t = unpack(z_t)
        eps_hat = net(x_t, y_fn.t).sample
        sigma_t = (1 - alphas_cumprod[y_fn.t]).sqrt()
        return pack(-eps_hat / sigma_t)

    def sigma2_fn():
        return (1 - alphas_cumprod[y_fn.t]).expand(config["n_samples"])

    def u_fn(y):
        t_frac = (y_fn.t.float() / (num_train_timesteps - 1)).expand(y.shape[0])
        return jac_net(y, t_frac)

    return (timesteps, alphas_cumprod, y_fn, base_step_with_t, step_scale_fn,
            score_at_zt_fn, sigma2_fn, u_fn)


# ---------------------------------------------------------------------------
# Guidance energies -- ambient + projected (same shape as the ENERGY_FACTORIES /
# PROJECTED_ENERGY_FACTORIES split in the SO(3)/toy notebooks).
# ---------------------------------------------------------------------------
def make_energy_factories(y_fn, sigma2_fn):
    def make_knn_energy_fn():
        def energy_fn(y):
            with torch.no_grad():
                neighbors, _ = basic_fr.ann_query(train_index, y, config["k_neighbors"])
            I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn())
            return I
        return energy_fn

    def make_knn_projected_energy_fn():
        return basic_fr.make_knn_projected_energy_fn(train_index, config["k_neighbors"], sigma2_fn)

    ENERGY_FACTORIES = {"basic_fr": make_knn_energy_fn}
    return ENERGY_FACTORIES, make_knn_projected_energy_fn


# ---------------------------------------------------------------------------
# Diagnostic: unclipped guidance gradient norm vs t, unprojected vs. projected (r'=max(R_PRIME_LIST)) --
# same dual-panel/color convention as the notebooks. Uses its OWN, much smaller step schedule
# (DIAG_N_STEPS) than the main sweep -- this walks the whole trajectory under no_grad but computes
# the (expensive) guidance gradient only every DIAG_STRIDE-th step.
# ---------------------------------------------------------------------------
def run_diagnostic():
    (timesteps, alphas_cumprod, y_fn, base_step_with_t, step_scale_fn,
     score_at_zt_fn, sigma2_fn, u_fn) = make_wiring(DIAG_N_STEPS)
    ENERGY_FACTORIES, make_knn_proj = make_energy_factories(y_fn, sigma2_fn)

    diag_r_prime = max(R_PRIME_LIST)
    diag_projector_fn = driver.make_lowrank_normal_projector_fn(u_fn, diag_r_prime)
    diag_energy_fns = {"basic_fr": ENERGY_FACTORIES["basic_fr"]()}
    diag_projected_energy_fns = {"basic_fr": make_knn_proj()}

    torch.manual_seed(config["seed"])
    z = pack(torch.randn(config["n_samples"], 3, 32, 32, device=config["device"]))

    rows = []
    for step_idx in range(1, DIAG_N_STEPS + 1):
        z_next = base_step_with_t(z, step_idx)
        if step_idx % DIAG_STRIDE == 0:
            t_val = timesteps[step_idx - 1].item()
            sigma2_val = (1 - alphas_cumprod[timesteps[step_idx - 1]]).item()
            for name, energy_fn in diag_energy_fns.items():
                grad, *_ = driver.guidance_grad(y_fn, z, energy_fn, target_range=None, grad_clip=None)
                gnorm = grad.norm(dim=-1)
                rows.append({"step_idx": step_idx, "t": t_val, "sigma2": sigma2_val, "method": name,
                             "projected": False, "grad_norm_mean": gnorm.mean().item()})
            for name, energy_fn in diag_projected_energy_fns.items():
                grad, *_ = driver.projected_guidance_grad(y_fn, z, energy_fn, diag_projector_fn,
                                                              target_range=None, grad_clip=None)
                gnorm = grad.norm(dim=-1)
                rows.append({"step_idx": step_idx, "t": t_val, "sigma2": sigma2_val, "method": name,
                             "projected": True, "grad_norm_mean": gnorm.mean().item()})
        z = z_next

    diag_df = pd.DataFrame(rows)
    UNPROJ_COLORS = {"basic_fr": "tab:blue"}
    PROJ_COLORS = {"basic_fr": "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True, sharey=True)
    for name, sub in diag_df[~diag_df["projected"]].groupby("method"):
        axes[0].plot(sub["t"], sub["grad_norm_mean"], label=name, color=UNPROJ_COLORS[name])
    for name, sub in diag_df[diag_df["projected"]].groupby("method"):
        axes[1].plot(sub["t"], sub["grad_norm_mean"], label=name, color=PROJ_COLORS[name])
    for ax, title in zip(axes, ["unprojected (ambient)", f"projected (JacNet, r'={diag_r_prime})"]):
        ax.set_xlabel("t (diffusion timestep)"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_title(title); ax.legend(fontsize=8)
    axes[0].set_ylabel("raw guidance grad norm (unclipped)")
    fig.suptitle("unclipped guidance gradient norm vs t")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "grad_norm_vs_t.png"), dpi=120)
    plt.close(fig)
    print(f"saved diagnostic plot to {OUT_DIR}/grad_norm_vs_t.png", flush=True)


# ---------------------------------------------------------------------------
# Metrics: mem_ratio (distance to training atoms) + fidelity_proxy (distance to held-out
# test images) -- both cheap, reusing assess_memorization_cifar10.py's own functions.
# ---------------------------------------------------------------------------
def evaluate(x_gen_flat):
    train_d1, train_d2, _ = nearest_neighbor_d1_d2(x_gen_flat, pack(X_train))
    mem_class = classify_images(x_gen_flat.shape[0], train_d1, train_d2, config["mem_ratio_threshold"])
    heldout_d1, _, _ = nearest_neighbor_d1_d2(x_gen_flat, X_heldout)
    return {"mem_ratio": mem_class["frac_memorized"], "fidelity_proxy": heldout_d1.mean().item()}


# ---------------------------------------------------------------------------
# Adaptive clip: cap the guidance correction at scale * ||s_theta(z_t, t)||, a per-sample,
# per-t cap that tracks the model's own drift magnitude, rather than one fixed constant for
# every (x, t) -- driver._resolve_clip_norm already accepts a callable adaptive_clip_fn(z_t),
# this is the first place actually supplying one.
# ---------------------------------------------------------------------------
def make_adaptive_clip_fn(score_at_zt_fn, scale):
    def adaptive_clip_fn(z_t):
        return scale * score_at_zt_fn(z_t).norm(dim=-1)
    return adaptive_clip_fn


def resolve_clip_spec(clip_str, score_at_zt_fn):
    """clip_str is either a plain number ("10") or "adaptive:<scale>" ("adaptive:1.0").
    Returns (grad_clip, is_adaptive) -- grad_clip is a float for the former, a callable
    adaptive_clip_fn(z_t) for the latter (both accepted transparently by
    driver._resolve_clip_norm)."""
    if clip_str.startswith("adaptive:"):
        scale = float(clip_str.split(":", 1)[1])
        return make_adaptive_clip_fn(score_at_zt_fn, scale), True
    return float(clip_str), False


# ---------------------------------------------------------------------------
# Full sweep: jacobian x r' x window x grad_clip x lam
# ---------------------------------------------------------------------------
def make_guidance_fn(y_fn, ENERGY_FACTORIES, make_knn_proj, u_fn, jacobian_name, r_prime, grad_clip):
    if jacobian_name == "none":
        energy_fn = ENERGY_FACTORIES["basic_fr"]()
        return driver.make_autograd_guidance_fn(y_fn, energy_fn, target_range=TARGET_RANGE, grad_clip=grad_clip)

    projector_fn = driver.make_lowrank_normal_projector_fn(u_fn, r_prime)
    energy_fn = make_knn_proj()
    return driver.make_projected_guidance_fn(y_fn, energy_fn, projector_fn, target_range=TARGET_RANGE, grad_clip=grad_clip)


def run_condition(base_step_with_t, n_steps, guidance_fn, lam, progress_lo, progress_hi, step_scale_fn, generator):
    torch.manual_seed(config["seed"])
    z_init = pack(torch.randn(config["n_samples"], 3, 32, 32, device=config["device"], generator=generator))
    if guidance_fn is None:
        z = z_init
        with torch.no_grad():
            for step_idx in range(1, n_steps + 1):
                z = base_step_with_t(z, step_idx)
        return unpack(z).clamp(-1, 1).detach()
    z_final, I_log = driver.guided_reverse_loop(
        z_init, n_steps, base_step_with_t, guidance_fn,
        progress_lo=progress_lo, progress_hi=progress_hi, ell=config["ell"], lam=lam,
        step_scale_fn=step_scale_fn)
    return unpack(z_final).clamp(-1, 1).detach()


def main():
    if os.environ.get("RUN_DIAGNOSTIC", "1") == "1":
        print("running gradient-norm diagnostic...", flush=True)
        run_diagnostic()

    n_steps = config["n_steps_eval"]
    gen = torch.Generator(device=config["device"]).manual_seed(SAMPLE_SEED)
    (timesteps, alphas_cumprod, y_fn, base_step_with_t, step_scale_fn,
     score_at_zt_fn, sigma2_fn, u_fn) = make_wiring(n_steps, generator=gen)
    ENERGY_FACTORIES, make_knn_proj = make_energy_factories(y_fn, sigma2_fn)

    print(f"unguided baseline ({config['n_samples']} samples, {n_steps} steps)...", flush=True)
    X_unguided = run_condition(base_step_with_t, n_steps, None, 0.0, 0.0, 1.0, step_scale_fn, gen)
    baseline = {"method": "unguided", "jacobian": "none", "r_prime": "n/a", "window": "n/a",
                "grad_clip": "n/a", "clip_mode": "n/a", "lam": 0.0, **evaluate(pack(X_unguided))}
    print(baseline, flush=True)

    rows = [baseline]
    JACOBIAN_NAMES = ["none", "jacnet"]
    n_jacobian_conditions = 1 + len(R_PRIME_LIST)
    n_combos = len(GRAD_CLIP_LIST) * len(WINDOW_LIST) * len(LAMBDA_LIST) * n_jacobian_conditions
    print(f"sweeping ~{n_combos} combos (jacobian x r' x window x grad_clip x lam)", flush=True)

    method = "basic_fr"
    for jacobian_name in JACOBIAN_NAMES:
        r_primes = ["n/a"] if jacobian_name == "none" else R_PRIME_LIST
        for r_prime in r_primes:
            for window_str in WINDOW_LIST:
                progress_lo, progress_hi = (float(v) for v in window_str.split("-"))
                for clip_str in GRAD_CLIP_LIST:
                    grad_clip, is_adaptive = resolve_clip_spec(clip_str, score_at_zt_fn)
                    guidance_fn = make_guidance_fn(
                        y_fn, ENERGY_FACTORIES, make_knn_proj, u_fn, jacobian_name,
                        r_prime if r_prime != "n/a" else None, grad_clip)
                    for lam_str in LAMBDA_LIST:
                        lam = float(lam_str)
                        t0 = time.time()
                        X = run_condition(base_step_with_t, n_steps, guidance_fn,
                                           lam, progress_lo, progress_hi, step_scale_fn, gen)
                        row = {"method": method, "jacobian": jacobian_name, "r_prime": r_prime,
                               "window": window_str, "grad_clip": clip_str,
                               "clip_mode": "adaptive" if is_adaptive else "fixed", "lam": lam,
                               **evaluate(pack(X))}
                        rows.append(row)
                        clip_tag = clip_str.replace(":", "")
                        tag = f"{method}_{jacobian_name}_r{r_prime}_win{window_str}_clip{clip_tag}_lam{lam_str}"
                        print(f"{tag}: {row['mem_ratio']=:.3f} {row['fidelity_proxy']=:.4f} "
                              f"({time.time() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["fidelity_rel"] = df["fidelity_proxy"] / baseline["fidelity_proxy"]
    df["mem_rel"] = df["mem_ratio"] / max(baseline["mem_ratio"], 1e-8)
    df["combined_error"] = df["fidelity_rel"] + df["mem_rel"]
    df.to_csv(os.path.join(OUT_DIR, "fr_sweep_results.csv"), index=False)
    print(f"{len(df)} configs run, saved to {OUT_DIR}/fr_sweep_results.csv", flush=True)

    guided = df[df["method"] != "unguided"]
    print("\nBest achievable combined_error per (method, jacobian):", flush=True)
    best_per_method = guided.loc[guided.groupby(["method", "jacobian"])["combined_error"].idxmin()]
    print(best_per_method[["method", "jacobian", "r_prime", "window", "grad_clip", "lam",
                            "mem_ratio", "fidelity_proxy", "combined_error"]].to_string(index=False), flush=True)

    # Per-parameter marginals (window, grad_clip, lam, r') -- same median (robustness) / min
    # (best-case) breakdown as the toy/SO(3) notebooks.
    PARAM_COLS = {"window": WINDOW_LIST, "grad_clip": GRAD_CLIP_LIST, "lam": [float(l) for l in LAMBDA_LIST]}
    for param_col, values in PARAM_COLS.items():
        stats = guided.groupby(["method", "jacobian", param_col])["combined_error"].agg(["median", "min"]).reset_index()
        best_median = stats.loc[stats.groupby(["method", "jacobian"])["median"].idxmin()]
        print(f"\n-- {param_col} -- best by median:", flush=True)
        print(best_median.sort_values(["method", "jacobian"]).to_string(index=False), flush=True)

    jacnet_only = guided[guided["jacobian"] == "jacnet"]
    r_stats = jacnet_only.groupby(["method", "r_prime"])["combined_error"].agg(["median", "min"]).reset_index()
    r_best_median = r_stats.loc[r_stats.groupby("method")["median"].idxmin()]
    print("\n-- r_prime -- (jacnet only) best by median:", flush=True)
    print(r_best_median.sort_values("method").to_string(index=False), flush=True)

    GROUP_COLORS = {
        ("basic_fr", "none"): "tab:blue", ("basic_fr", "jacnet"): "tab:green",
    }
    GROUPS = list(GROUP_COLORS.keys())
    JACNET_GROUPS = [g for g in GROUPS if g[1] == "jacnet"]

    def grouped_boxplot(ax, param_col, values, data, groups):
        width = 0.8 / len(groups)
        for gi, (method, jacobian) in enumerate(groups):
            sub = data[(data.method == method) & (data.jacobian == jacobian)]
            positions = [i + (gi - (len(groups) - 1) / 2) * width for i in range(len(values))]
            box_data = [sub.loc[sub[param_col] == v, "combined_error"].values for v in values]
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(GROUP_COLORS[(method, jacobian)]); patch.set_alpha(0.6)
        ax.set_xticks(range(len(values))); ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right")
        ax.set_yscale("log"); ax.set_title(f"combined_error by {param_col}")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, (param_col, values) in zip(axes[:3], PARAM_COLS.items()):
        grouped_boxplot(ax, param_col, values, guided, GROUPS)
    grouped_boxplot(axes[3], "r_prime", R_PRIME_LIST, jacnet_only, JACNET_GROUPS)
    axes[0].set_ylabel("combined_error (log scale)")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6) for c in GROUP_COLORS.values()]
    legend_labels = [f"{m}/{j}" for m, j in GROUPS]
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("combined_error marginal distribution per parameter (lower is better)", y=1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "param_marginals.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved marginals plot to {OUT_DIR}/param_marginals.png", flush=True)


if __name__ == "__main__":
    main()
