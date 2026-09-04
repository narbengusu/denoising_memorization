"""Figure 5 -- how the guidance window and temperature setpoint are chosen.

One unguided reverse trajectory per characterized CelebA-HQ seed, tracking: (a) untempered
k-NN Fisher-Rao energy I(D_t(x)), (b) dominant responsibility q_max, untempered vs. at
setpoint q*, (c) guidance gradient norm, untempered vs. tempered (shows the dead zone and
that the setpoint removes it), (d) latent-space d1/d2 (how fast the memorized match locks
in). Measured on the unguided path, before any intervention.

Writes out/fig5.{pdf,png}, caches traces in cache/fig5_traces.pt.
"""
import os, sys, gc

import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "workshop_figures"))

from guidance import driver, basic_fr
import wsstyle

torch.backends.cuda.enable_cudnn_sdp(False)

mpl_fs = 15
wsstyle.use_style(font_size=mpl_fs)

EXP = os.path.join(REPO, "experiments", "celeba_hq")
CACHE = os.path.join(REPO, "workshop_figures", "cache", "fig5_traces.pt")

device = os.environ.get("DEVICE") or (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu"))

config = {
    "ckpt_path": os.path.join(EXP, "checkpoints", "celeba_hq_edm_finetuned.pt"),
    "latents_path": os.path.join(EXP, "checkpoints", "celeba_hq_latents.pt"),
    "finetune_idx_path": os.path.join(EXP, "checkpoints", "celeba_hq_finetune_train_idx.npy"),
    "n_steps": 32, "sigma_min": 0.002, "sigma_max": 80.0, "rho": 7.0,
    "k_neighbors": 15, "mem_ratio_threshold": 1 / 3,
}
SEEDS = [22, 20, 21, 11, 15]
Q_TARGET = 0.8            # the setpoint used for the reported CelebA runs
WINDOW = (0.35, 1.0)      # the window it justifies
N_STEPS = config["n_steps"]


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 4, 32, 32)


def build():
    ckpt = torch.load(config["ckpt_path"], map_location="cpu")
    sigma_data = ckpt["config"]["sigma_data"]
    ema_state = ckpt["model_ema"]
    del ckpt
    gc.collect()
    net = UNet2DModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 384, 384),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    net.load_state_dict(ema_state)
    del ema_state
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    blob = torch.load(config["latents_path"], map_location="cpu")
    all_latents = blob["latents"].float()
    ft_idx = np.load(config["finetune_idx_path"])
    X_train_flat = pack(all_latents[ft_idx].to(device))
    del blob, all_latents
    gc.collect()
    return net, sigma_data, basic_fr.build_flat_index(X_train_flat)


def make_sigmas():
    step_idx = torch.arange(N_STEPS, dtype=torch.float64)
    s = (config["sigma_max"] ** (1 / config["rho"]) + step_idx / (N_STEPS - 1) *
         (config["sigma_min"] ** (1 / config["rho"]) - config["sigma_max"] ** (1 / config["rho"]))
         ) ** config["rho"]
    return torch.cat([s, torch.zeros(1)]).float().to(device)


def trace(seed, net, sigma_data, train_index):
    sigmas = make_sigmas()

    def edm_precond(x, sigma):
        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()
        F_x = net(c_in * x, 0.25 * sigma.log().flatten()).sample
        return c_skip * x + c_out * F_x

    def y_fn(z_t):
        return pack(edm_precond(unpack(z_t), y_fn.s.expand(1)))

    def base_step(z_t, step_idx):
        s_cur, s_next = sigmas[step_idx - 1], sigmas[step_idx]
        x_t = unpack(z_t)
        d_cur = (x_t - edm_precond(x_t, s_cur.expand(1))) / s_cur
        x_next = x_t + (s_next - s_cur) * d_cur
        if s_next > 0:
            d_next = (x_next - edm_precond(x_next, s_next.expand(1))) / s_next
            x_next = x_t + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        return pack(x_next)

    def sigma2_fn():
        return (y_fn.s ** 2).expand(1)

    def energy_fn_for(q_target):
        def energy_fn(y):
            with torch.no_grad():
                neighbors, sq_d = basic_fr.ann_query(train_index, y, config["k_neighbors"])
            bf = basic_fr.beta_soft_for(sq_d, q_target)
            I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn(), beta_soft=bf)
            return I
        return energy_fn

    gen = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)
    z = pack(torch.randn(1, 4, 32, 32, generator=gen).to(device) * sigmas[0])

    rec = {key: [] for key in ("I", "I_temp", "q_max", "q_max_temp", "gn_raw", "gn_temp",
                                "ratio", "step_norm", "beta_soft", "sigma")}
    for step_idx in range(1, N_STEPS + 1):
        y_fn.s = sigmas[step_idx - 1]
        with torch.no_grad():
            y = y_fn(z)
            neighbors, sq_d = basic_fr.ann_query(train_index, y, config["k_neighbors"])
            I, q, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn())
            bf = basic_fr.beta_soft_for(sq_d, Q_TARGET)
            I_t, q_t, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn(), beta_soft=bf)
            rec["I"].append(I.item())
            rec["I_temp"].append(I_t.item())
            rec["beta_soft"].append(bf.item())
            rec["sigma"].append(y_fn.s.item())
            rec["q_max"].append(q.max(-1).values.item())
            rec["q_max_temp"].append(q_t.max(-1).values.item())
            rec["ratio"].append((sq_d[0, 0] / sq_d[0, 1]).sqrt().item())

        g_raw, _, _ = driver.guidance_grad(y_fn, z, energy_fn_for(None))
        g_temp, _, _ = driver.guidance_grad(y_fn, z, energy_fn_for(Q_TARGET))
        rec["gn_raw"].append(g_raw.norm().item())
        rec["gn_temp"].append(g_temp.norm().item())

        with torch.no_grad():
            z_next = base_step(z, step_idx)
        rec["step_norm"].append((z_next - z).norm().item())
        z = z_next

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["progress"] = np.arange(1, N_STEPS + 1) / N_STEPS
    out["seed"] = seed
    return out


def compute():
    if os.path.exists(CACHE):
        traces = torch.load(CACHE, weights_only=False)
        if [t["seed"] for t in traces] == SEEDS:
            print(f"reusing {CACHE}")
            return traces
    net, sigma_data, train_index = build()
    traces = []
    for s in SEEDS:
        traces.append(trace(s, net, sigma_data, train_index))
        print(f"seed {s}: done")
    torch.save(traces, CACHE)
    return traces


def render(traces):
    """One legend outside every axes, short headers over a rule grid, no per-panel legends."""
    palette = [wsstyle.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange", "purple")]
    # log-axis floors: a curve flat at the floor means the underlying value is exactly zero (dead zone)
    I_FLOOR = 1e-8
    GRAD_FLOOR = 1e-18
    DASH = (0, (2.2, 1.8))
    WINDOW_ALPHA = 0.16

    HEADERS = ["FR energy", "responsibility", "gradient norm", "NN ratio"]

    fig = plt.figure(figsize=(19.0, 5.6))
    axs, blocks, block_rects = wsstyle.place_grid(
        fig, n_rows=1, block_sizes=[1] * 4,
        left=0.050, right=0.010, top=0.815, bottom=0.250,
        block_gap=0.052, panel_gap=0.0, row_gap=0.0)
    ax = [axs[0][j] for j in range(4)]

    for a in ax:
        a.axvspan(*WINDOW, color=wsstyle.C["field"], alpha=WINDOW_ALPHA, lw=0, zorder=0)
        a.set_xlabel("sampling progress")
        a.set_xlim(0, 1)
        wsstyle.grid_on(a)

    for t, c in zip(traces, palette):
        p = t["progress"]
        ax[0].plot(p, np.maximum(np.abs(t["I"]), I_FLOOR), lw=2.4, color=c)
        ax[0].plot(p, np.maximum(np.abs(t["I_temp"]), I_FLOOR), lw=2.4, color=c, ls=DASH)
        ax[1].plot(p, t["q_max"], lw=2.4, color=c)
        ax[1].plot(p, t["q_max_temp"], lw=2.4, color=c, ls=DASH)
        ax[2].plot(p, np.maximum(t["gn_raw"], GRAD_FLOOR), lw=2.4, color=c)
        ax[2].plot(p, np.maximum(t["gn_temp"], GRAD_FLOOR), lw=2.4, color=c, ls=DASH)
        ax[3].plot(p, t["ratio"], lw=2.4, color=c)

    ax[0].set_yscale("log")
    ax[0].set_ylim(I_FLOOR / 3, None)
    ax[0].set_ylabel(r"$|\widetilde{\mathcal{I}}_t(D_t(x_t))|$")

    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel(r"$q_{\max}$")

    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$\|\nabla_{x_t}\widetilde{\mathcal{I}}_t\|$")

    ax[3].set_ylim(0, 1.05)
    ax[3].set_ylabel(r"$d_1/d_2$")

    rule_ys = []
    for head, block, rect in zip(HEADERS, blocks, block_rects):
        flat = wsstyle.flatten(block)
        y = wsstyle.block_header(fig, flat, head, x_extent=rect, dy=0.058)
        wsstyle.header_rule(fig, rect, y - 0.014)
        rule_ys.append(y - 0.014)

    y_bottom = min(a.get_position().y0 for a in ax) - 0.150
    for x in wsstyle.gutter_positions(block_rects):
        wsstyle.separator(fig, x, y_bottom, max(rule_ys), color="#C4C4C4")

    entries = [(f"seed {t['seed']}", "line", c) for t, c in zip(traces, palette)]
    entries += [(r"$\beta_t = \sigma_t^2$", "line", "#3A3A3A"),
                (rf"$\beta_t$ at $q^\star = {Q_TARGET:g}$", "line", "#3A3A3A", DASH),
                ("proposed guidance window", "patch",
                 wsstyle.blend_on_white(wsstyle.C["field"], WINDOW_ALPHA))]
    wsstyle.figure_legend(fig, entries, ncol=len(entries), y=0.140, loc="upper center")

    return wsstyle.save(fig, "fig5")


if __name__ == "__main__":
    print(render(compute()))
