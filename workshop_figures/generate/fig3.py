#!/usr/bin/env python3
"""Figure 3 -- Fisher-Rao fields across bandwidths.

h^2 * I_{N,h}(x) painted on the surface for three bandwidths, with the predicted transition
curve Lambda_{N,h}(x)=1 in black. Standalone, no repo deps beyond wsstyle.
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
import contourpy

from _surface_common import f, gamma, N_CL, N_UN, CTR, SD, Y, N, OBS_COLOR, clean3d, OUT_DIR


def fisher_field(h, n):
    U, V = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    P = np.column_stack([U.ravel(), V.ravel(), f(U, V).ravel()])
    out = np.empty(len(P))
    Y2 = (Y ** 2).sum(1)
    for i in range(0, len(P), 3000):
        X = P[i:i + 3000]
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        lg = -d2 / (2 * h * h)
        lg -= lg.max(1, keepdims=True)
        q = np.exp(lg)
        q /= q.sum(1, keepdims=True)
        mu = q @ Y
        out[i:i + 3000] = np.maximum((q * Y2[None, :]).sum(1) - (mu ** 2).sum(1), 0.0) / h ** 2
    return U, V, f(U, V), out.reshape(n, n)


def density_area(U, V):
    """Sampling density with respect to surface area (mixture of truncated normal and uniform in (u,v))."""
    g = np.exp(-((U - CTR[0]) ** 2 + (V - CTR[1]) ** 2) / (2 * SD ** 2))
    Uf, Vf = np.meshgrid(np.linspace(0.02, 0.98, 600), np.linspace(0.02, 0.98, 600))
    gf = np.exp(-((Uf - CTR[0]) ** 2 + (Vf - CTR[1]) ** 2) / (2 * SD ** 2))
    Z = gf.mean() * 0.96 ** 2                       # normalisation of the truncated normal
    rho_uv = (N_CL / N) * g / Z + (N_UN / N) / 0.96 ** 2
    eps = 1e-4
    fu = (f(U + eps, V) - f(U - eps, V)) / (2 * eps)
    fv = (f(U, V + eps) - f(U, V - eps)) / (2 * eps)
    return rho_uv / np.sqrt(1 + fu ** 2 + fv ** 2)


field_stats = []     # (h, fraction of surface below the resolution threshold), for the caption


def field_figure(fname, hs=(0.12, 0.045, 0.015), n=300):
    fig = plt.figure(figsize=(16, 6.3))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05], left=0.015, right=0.985,
                           top=0.93, bottom=0.10, wspace=0.02, hspace=0.0)
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.5)
    cmap = plt.get_cmap("RdBu_r")
    U, V = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    rho = density_area(U, V)
    for j, h in enumerate(hs):
        ax = fig.add_subplot(gs[0, j], projection="3d"); clean3d(ax)
        U, V, Z, Fh = fisher_field(h, n)
        col = cmap(norm(np.log10(np.clip(Fh / 2.0, 10 ** -1.5, 10 ** 1.5))))
        ax.plot_surface(U, V, Z, facecolors=col, rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, zorder=1, rasterized=True)
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2] + 0.004, s=2.5, color=OBS_COLOR, depthshade=False, zorder=3)
        # predicted transition: Λ_{N,h}(x) = N rho(x) v_2 h^2 = 1
        lam = N * rho * np.pi * h ** 2
        for c in contourpy.contour_generator(U, V, lam).lines(1.0):
            ax.plot(c[:, 0], c[:, 1], f(c[:, 0], c[:, 1]) + 0.006, color="k", lw=1.6, zorder=4)
        frac = (lam < 1).mean()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(-0.5, 0.7)
        ax.set_box_aspect((1, 1, 1.0), zoom=1.45)
        ax.view_init(elev=52, azim=-60)
        field_stats.append((h, frac))
        ax.set_title(rf"$h = {h}$", fontsize=14, pad=2)
    cax = fig.add_axes([0.30, 0.075, 0.40, 0.025])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=[-1.5, -1, 0, 1, 1.5],
                                          orientation="horizontal")
    cb.set_ticklabels([r"$\leq 10^{-1.5}$", r"$10^{-1}$", r"$1$", r"$10$", r"$\geq 10^{1.5}$"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label(r"$h^2\,\hat{\mathcal{I}}_{N,h}(x)\,/\,m$", fontsize=13, labelpad=4)
    fig.savefig(fname + ".png", dpi=200)
    fig.savefig(fname + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    field_figure(os.path.join(OUT_DIR, "fig3"))

    print()
    print("Figure 3 caption numbers")
    for h, frac in field_stats:
        print(f"    h={h:<6} Lambda_Nh(x) < 1 on {100 * frac:.0f}% of the surface")
    print()
    print(f"wrote -> {os.path.normpath(OUT_DIR)}/fig3.{{pdf,png}}")
