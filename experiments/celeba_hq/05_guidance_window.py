"""Supplementary figure: FR energy, q_max, guidance gradient norm and d1/d2 along unguided CelebA-HQ trajectories.
Writes all_figures/supp_guidance_window.pdf; caches traces in cache/guidance_window_traces.pt."""
import os, sys

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from guidance import driver, basic_fr
from models import celeba_hq_edm as E
from plotting import wsstyle

wsstyle.large_style()

CACHE = os.path.join(HERE, "cache", "guidance_window_traces.pt")
device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
SEEDS = [22, 20, 21, 11, 15]
Q_TARGET = E.FR_SETTINGS["q_target"]
WINDOW = (E.FR_SETTINGS["window_lo"], E.FR_SETTINGS["window_hi"])
K = E.FR_SETTINGS["k_neighbors"]
N_STEPS = E.N_STEPS


def trace(seed, m):
    sigmas = m["sigmas"]

    def y_fn(z_t):
        return E.pack(E.precond(m, E.unpack(z_t), y_fn.s.expand(1)))

    def base_step(z_t, step_idx):
        x_t = E.unpack(z_t)
        return E.pack(E.heun_step(m, x_t, step_idx - 1, E.precond(m, x_t, sigmas[step_idx - 1].expand(1))))

    def sigma2_fn():
        return (y_fn.s ** 2).expand(1)

    def energy_fn(y):
        with torch.no_grad():
            neighbors, _ = basic_fr.ann_query(m["train_index"], y, K)
        I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn())
        return I

    z = E.pack(E.z_init(m, seed))
    rec = {key: [] for key in ("I", "q_max", "gn_raw", "ratio", "step_norm", "beta_soft", "sigma")}
    for step_idx in range(1, N_STEPS + 1):
        y_fn.s = sigmas[step_idx - 1]
        with torch.no_grad():
            y = y_fn(z)
            neighbors, sq_d = basic_fr.ann_query(m["train_index"], y, K)
            I, q, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2_fn())
            bf = basic_fr.beta_soft_for(sq_d, Q_TARGET)
            rec["I"].append(I.item())
            rec["beta_soft"].append(bf.item())
            rec["sigma"].append(y_fn.s.item())
            rec["q_max"].append(q.max(-1).values.item())
            rec["ratio"].append((sq_d[0, 0] / sq_d[0, 1]).sqrt().item())

        g_raw, _, _ = driver.guidance_grad(y_fn, z, energy_fn)
        rec["gn_raw"].append(g_raw.norm().item())

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
    m = E.load(device, with_vae=False)
    traces = []
    for s in SEEDS:
        traces.append(trace(s, m))
        print(f"seed {s}: done")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(traces, CACHE)
    return traces


def render(traces):
    palette = [wsstyle.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange", "purple")]
    # a curve flat at the floor is exactly zero
    I_FLOOR = 1e-8
    GRAD_FLOOR = 1e-18
    WINDOW_ALPHA = 0.16

    fig, ax = plt.subplots(1, 4, figsize=(wsstyle.FULL_WIDTH, 4.9))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.36, wspace=0.42)

    # x axis is sampling progress in [0, 1], the same coordinate the guidance window is defined in
    for a in ax:
        a.axvspan(WINDOW[0], WINDOW[1], color=wsstyle.C["field"], alpha=WINDOW_ALPHA, lw=0, zorder=0)
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1])
        a.set_xlabel("sampling progress")
        wsstyle.grid_on(a)

    for t, c in zip(traces, palette):
        s = t["progress"]
        ax[0].plot(s, np.maximum(np.abs(t["I"]), I_FLOOR), lw=2.4, color=c)
        ax[1].plot(s, t["q_max"], lw=2.4, color=c)
        ax[2].plot(s, np.maximum(t["gn_raw"], GRAD_FLOOR), lw=2.4, color=c)
        ax[3].plot(s, t["ratio"], lw=2.4, color=c)

    ax[0].set_yscale("log")
    ax[0].set_ylim(I_FLOOR / 3, None)
    ax[0].set_ylabel(r"$\mathcal{I}_t$")

    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel(r"$q_{\max}$")

    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"$\|\nabla_{x_t}\mathcal{I}_t\|$")

    ax[3].set_ylim(0, 1.05)
    ax[3].set_ylabel(r"$d_1/d_2$")

    entries = [(f"seed {t['seed']}", "line", c) for t, c in zip(traces, palette)]
    entries += [("proposed guidance window", "patch",
                 wsstyle.blend_on_white(wsstyle.C["field"], WINDOW_ALPHA))]
    wsstyle.figure_legend(fig, entries, ncol=len(entries), y=0.08, loc="upper center")

    return wsstyle.save(fig, "supp_guidance_window")


if __name__ == "__main__":
    print(render(compute()))
