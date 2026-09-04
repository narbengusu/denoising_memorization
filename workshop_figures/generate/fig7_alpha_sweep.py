"""Bayesian FR dirichlet_alpha sweep for fig7.ipynb.

Runs the bayesian arm only, for each alpha in ALPHAS, reusing the notebook's cached
unguided/basic runs. Renders one figure per alpha to out/fig7_alpha{alpha}.{png,pdf}.

Usage: .venv/bin/python workshop_figures/generate/fig7_alpha_sweep.py
"""
import os, sys, time

os.environ.setdefault("DEVICE", "cpu")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "cifar10"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "workshop_figures"))

from diffusers import UNet2DModel, DDPMScheduler
from cifar10_data import load_cifar10
from assess_memorization_cifar10 import nearest_neighbor_d1_d2
from guidance import driver, basic_fr, bayesian_fr
import wsstyle

mpl_fs = 19
wsstyle.use_style(font_size=mpl_fs)

ALPHAS = [0.75, 1.0, 2.0, 0.4]

EXP = os.path.join(REPO, "experiments", "cifar10")
CACHE = os.path.join(REPO, "workshop_figures", "cache", "fig7_runs.pt")

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.path.join(EXP, "pretrained", "ddpm-cifar10-32"),
    "score_ckpt_path": os.path.join(EXP, "trained_data", "ddpm_finetuned.pt"),
    "single_sample_path": os.path.join(EXP, "checkpoints", "single_sample.pt"),
    "cifar10_data_dir": os.path.join(REPO, "data", "cifar-10-batches-py"),
    "finetune_seed": 0,
    "n_train": 1000,
    "mem_ratio_threshold": 1 / 2,
    "seeds": [9, 12, 13, 18, 19],
    "k_neighbors": 15,
    "ell": 1,
}

DEFAULTS = {"window_lo": 0.45, "window_hi": 1.0, "eta": 0.1, "q_target": 0.8}
PER_SEED = {
     9: {}, 12: {}, 13: {"window_lo": 0.40}, 18: {}, 19: {},
}

B_BOOTSTRAP = 256
METHODS = ["unguided", "basic", "bayesian"]
METHOD_LABELS = {"unguided": "Unguided", "basic": "FR", "bayesian": "Bayesian FR"}


def settings_for(seed):
    return {**DEFAULTS, **PER_SEED.get(seed, {})}


def _run_key(seed, method, st, alpha, B, k):
    if method == "unguided":
        return f"unguided|seed={seed}"
    common = (f"seed={seed}|lo={st['window_lo']}|hi={st['window_hi']}"
              f"|eta={st['eta']}|qt={st['q_target']}|k={k}")
    if method == "basic":
        return f"basic|{common}"
    return f"bayesian|{common}|alpha={alpha}|B={B}"


def run_key(seed, method, alpha):
    return _run_key(seed, method, settings_for(seed), alpha, B_BOOTSTRAP, config["k_neighbors"])


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 3, 32, 32)


print("loading data/model...")
single_sample = torch.load(config["single_sample_path"], map_location="cpu", weights_only=False)
assert single_sample["device"] == config["device"], (
    f"candidates were generated on {single_sample['device']!r}, this run is on "
    f"{config['device']!r} -- set DEVICE to match or the seeds will not reproduce")

images, _ = load_cifar10(data_dir=config["cifar10_data_dir"], split="train")
rng = np.random.default_rng(config["finetune_seed"])
train_idx = rng.choice(len(images), size=config["n_train"], replace=False)
X_train = images[train_idx].to(config["device"])
X_train_flat = pack(X_train)
train_index = basic_fr.build_flat_index(X_train_flat)

net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
score_ckpt = torch.load(config["score_ckpt_path"], map_location=config["device"], weights_only=False)
net.load_state_dict(score_ckpt.get("model_ema", score_ckpt["model"]))
net.eval()
for p in net.parameters():
    p.requires_grad_(False)

scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
alphas_cumprod = scheduler.alphas_cumprod.to(config["device"])

_by_seed = {c["seed"]: c for c in single_sample["candidates"]}
missing = [s for s in config["seeds"] if s not in _by_seed]
assert not missing, f"seeds {missing} missing from {config['single_sample_path']}"


def select_seed(seed):
    global candidate, n_steps
    candidate = _by_seed[seed]
    n_steps = candidate["n_steps"]
    scheduler.set_timesteps(n_steps)
    return candidate


def make_wiring(generator):
    def denoise_to_y(z_t, t):
        x_t = unpack(z_t)
        eps_hat = net(x_t, t).sample
        a_t = alphas_cumprod[t]
        y = (x_t - (1 - a_t).sqrt() * eps_hat) / a_t.sqrt()
        return pack(y).clamp(-1, 1)

    def y_fn(z_t):
        return denoise_to_y(z_t, y_fn.t)

    def base_step_with_t(z_t, step_idx):
        t = scheduler.timesteps[step_idx - 1]
        y_fn.t = t
        x_t = unpack(z_t)
        eps_hat = net(x_t, t).sample
        return pack(scheduler.step(eps_hat, t, x_t, generator=generator).prev_sample)

    def sigma2_fn():
        return (1 - alphas_cumprod[y_fn.t]).expand(1)

    return y_fn, base_step_with_t, sigma2_fn


def make_energy_fn(method, sigma2_fn, q_target, alpha):
    def energy_fn(y):
        with torch.no_grad():
            neighbors, sq_d = basic_fr.ann_query(train_index, y, config["k_neighbors"])
        bf = basic_fr.beta_soft_for(sq_d, q_target)
        if method == "basic":
            I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn(), beta_soft=bf)
        elif method == "bayesian":
            I, _, _ = bayesian_fr.fisher_rao_energy_bb(
                y, neighbors, sigma2_fn(), B=B_BOOTSTRAP,
                dirichlet_alpha=alpha, beta_soft=bf)
        else:
            raise ValueError(method)
        return I
    return energy_fn


def evaluate(x_final):
    d1, d2, nn_idx = nearest_neighbor_d1_d2(pack(x_final), X_train_flat)
    ratio = (d1 / d2).item()
    return {"d1": d1.item(), "d2": d2.item(), "ratio": ratio,
            "nn_idx": int(nn_idx.item()),
            "is_memorized": ratio < config["mem_ratio_threshold"]}


def run_condition(seed, method, alpha):
    st = settings_for(seed)
    gen = torch.Generator(device=config["device"]).manual_seed(seed)
    y_fn, base_step_with_t, sigma2_fn = make_wiring(gen)
    z_init = pack(torch.randn(1, 3, 32, 32, device=config["device"], generator=gen))

    if method == "unguided":
        z = z_init
        with torch.no_grad():
            for step_idx in range(1, n_steps + 1):
                z = base_step_with_t(z, step_idx)
    else:
        guidance_fn = driver.make_autograd_guidance_fn(
            y_fn, make_energy_fn(method, sigma2_fn, st["q_target"], alpha), target_range=None)
        z, _ = driver.guided_reverse_loop(
            z_init, n_steps, base_step_with_t, guidance_fn,
            progress_lo=st["window_lo"], progress_hi=st["window_hi"],
            ell=config["ell"], trust_region=st["eta"])

    x_final = unpack(z).clamp(-1, 1)
    return x_final.detach().cpu(), evaluate(x_final)


def load_store():
    blob = torch.load(CACHE, weights_only=False) if os.path.exists(CACHE) else {}
    return blob.get("runs", {})


def save_store(store):
    torch.save({"runs": store}, CACHE)


def to_img(x):
    t = x.squeeze(0) if x.dim() == 4 else x
    return ((t.permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()


def render(alpha, store):
    payload_rows = {seed: {"settings": settings_for(seed),
                            "methods": {m: store[run_key(seed, m, alpha)] for m in METHODS}}
                     for seed in config["seeds"]}

    print(f"\ncaption numbers  (Bayesian FR: dirichlet_alpha={alpha}, B={B_BOOTSTRAP})")
    for seed in config["seeds"]:
        r = payload_rows[seed]
        bits = "  ".join(f"{m}: ratio={r['methods'][m]['stats']['ratio']:.3f} "
                         f"mem={r['methods'][m]['stats']['is_memorized']}" for m in METHODS)
        print(f"  seed {seed:>3} (window_lo={r['settings']['window_lo']}): {bits}")

    SEEDS = config["seeds"]
    BLOCKS = [(METHOD_LABELS[m], m) for m in METHODS]
    COL_TITLES = ["Generated", "Nearest train"]

    fig = plt.figure(figsize=(15.0, 13.0))
    axes, blocks, block_rects = wsstyle.place_grid(
        fig, n_rows=len(SEEDS), block_sizes=[2] * len(BLOCKS),
        left=0.058, right=0.012, top=0.880, bottom=0.030,
        block_gap=0.050, panel_gap=0.012, row_gap=0.020)

    for i, seed in enumerate(SEEDS):
        row = payload_rows[seed]
        for b, (_, method) in enumerate(BLOCKS):
            cell = row["methods"][method]
            for j, img in enumerate([to_img(cell["x"]), to_img(cell["nn"])]):
                ax = blocks[b][i][j]
                ax.imshow(img, interpolation="nearest")
                wsstyle.strip_axes(ax)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(1.0)
                    sp.set_edgecolor("#7A7A7A" if j == 0 else "#D0D0D0")

    for i, seed in enumerate(SEEDS):
        axes[i][0].set_ylabel(f"seed {seed}", labelpad=14)

    for b in range(len(BLOCKS)):
        for ax, t in zip(blocks[b][0], COL_TITLES):
            ax.set_title(t, fontsize=mpl_fs - 3, pad=8)

    rule_ys = []
    for (label, _), block, rect in zip(BLOCKS, blocks, block_rects):
        flat = wsstyle.flatten(block)
        y = wsstyle.block_header(fig, flat, label, x_extent=rect, dy=0.052)
        wsstyle.header_rule(fig, rect, y - 0.010)
        rule_ys.append(y - 0.010)

    GUTTER_COLOR = "#C4C4C4"
    y_top = max(rule_ys)
    y_bottom = min(ax.get_position().y0 for ax in wsstyle.flatten(axes)) - 0.012
    for x in wsstyle.gutter_positions(block_rects):
        wsstyle.separator(fig, x, y_bottom, y_top, color=GUTTER_COLOR)

    alpha_tag = str(alpha).replace(".", "p")
    paths = wsstyle.save(fig, f"fig7_alpha{alpha_tag}")
    plt.close(fig)
    print("saved ->", paths)


def main():
    for s in config["seeds"]:
        print(f"seed {s:>3}: {settings_for(s)}")

    store = load_store()
    print(f"cache holds {len(store)} runs")

    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha} ===")
        todo = [(seed, "bayesian") for seed in config["seeds"]
                if run_key(seed, "bayesian", alpha) not in store]
        print(f"{len(config['seeds']) - len(todo)} reused, {len(todo)} to run")

        for seed, method in todo:
            select_seed(seed)
            t0 = time.time()
            x_final, stats = run_condition(seed, method, alpha)
            store[run_key(seed, method, alpha)] = {
                "seed": seed, "method": method, "x": x_final,
                "nn": X_train[stats["nn_idx"]].detach().cpu(), "stats": stats,
                "settings": settings_for(seed),
                "alpha": alpha,
            }
            save_store(store)
            print(f"seed={seed:>3} {method:<9} alpha={alpha} ratio={stats['ratio']:.3f} "
                  f"memorized={str(stats['is_memorized']):<5} nn={stats['nn_idx']:<4} "
                  f"({time.time() - t0:.0f}s)")

        render(alpha, store)

    print("\nall alphas done:", ALPHAS)


if __name__ == "__main__":
    main()
