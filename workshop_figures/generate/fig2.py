#!/usr/bin/env python3
"""Figure 2 -- finite-sample resolution.

Zoom chain: overview + two rows (dense/sparse query point) magnifying
B(x,h_1) -> B(x,h_2) -> B(x,h_3) on a curved surface. Standalone, no repo deps beyond wsstyle.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

from _surface_common import (
    f, gamma, N_CL, N_UN, CTR, SD, UV, Y, N, QUERIES, RADII, ROW_COLORS,
    OBS_COLOR, surface_patch, sphere_mesh, draw_ball, cap_boundary, draw_ring,
    clean3d, clean, screen_points, responsibilities, OUT_DIR,
)
import wsstyle


def zoom_chain(fname):
    fig = plt.figure(figsize=(15.5, 7.4))
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1.75, 1, 1, 1],
                           height_ratios=[1.0, 0.20, 1.0],
                           left=0.01, right=0.985, top=0.94, bottom=0.03, wspace=0.10, hspace=0.05)
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[2.6, 1.0], hspace=0.0)

    # ---- overview
    ax0 = fig.add_subplot(gs0[0], projection="3d")
    clean3d(ax0)
    surface_patch(ax0, -0.12, 1.12, -0.12, 1.12, n=120)
    ax0.scatter(Y[:, 0], Y[:, 1], Y[:, 2] + 0.003, s=9, color=OBS_COLOR,
                edgecolors="0.25", linewidths=0.3, depthshade=False, zorder=4)
    for name, uv in QUERIES.items():
        col = ROW_COLORS[name]
        x = gamma(uv)[0]
        draw_ball(ax0, x, RADII[0], col, face_alpha=0.14, zorder=6)
        draw_ring(ax0, cap_boundary(x, RADII[0], -0.15, 1.15, -0.15, 1.15), col, lw=1.6, ls=(0, (4, 3)), zorder=7)
        ax0.scatter(*x, s=60, facecolor="white", edgecolor=col, lw=1.8, depthshade=False, zorder=8)
    ax0.set_xlim(-0.1, 1.1); ax0.set_ylim(-0.1, 1.1); ax0.set_zlim(-0.45, 0.75)
    ax0.set_box_aspect((1.2, 1.2, 1.2), zoom=1.6)
    ax0.view_init(elev=38, azim=-58)

    # ---- legend block
    axl = fig.add_subplot(gs0[1]); clean(axl)
    handles = [Rectangle((0, 0), 1, 1, facecolor=(0.35, 0.45, 0.6, 0.25), edgecolor=(0.35, 0.45, 0.6, 0.6)),
               Line2D([], [], color="0.3", lw=1.4, ls=(0, (4, 3))),
               Line2D([], [], color="0.3", lw=1.6),
               Line2D([], [], marker="o", color="none", markerfacecolor=OBS_COLOR, markeredgecolor="0.25", markersize=9),
               Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="0.3", markeredgewidth=1.8, markersize=8)]
    # symbols only; glosses (kernel std, ball volume, responsibility) are caption material
    labels = [r"$\mathbb{B}(x,h)$",
              r"$\partial\mathbb{B}(x,h)\cap\mathcal{K}$",
              r"next panel's $h$",
              r"$\hat{y}_i$",
              r"$x$"]
    axl.legend(handles, labels, loc="upper left", fontsize=12.5, frameon=False, handlelength=1.8,
               borderaxespad=0.0, labelspacing=0.55)

    # ---- rows
    chain = []          # (source axes, 3-D points whose silhouette we connect from, destination axes, colour)
    for r, (name, uv) in enumerate(QUERIES.items()):
        col = ROW_COLORS[name]
        x = gamma(uv)[0]
        base = 2 * r
        prev_ax = ax0
        for k, h in enumerate(RADII):
            axz = fig.add_subplot(gs[base, k + 1], projection="3d"); clean3d(axz)
            L = 1.35 * h
            surface_patch(axz, x[0] - 1.1 * L, x[0] + 1.1 * L, x[1] - 1.1 * L, x[1] + 1.1 * L, n=70)
            draw_ball(axz, x, h, col, face_alpha=0.10, zorder=2)
            draw_ring(axz, cap_boundary(x, h, x[0] - 1.2 * L, x[0] + 1.2 * L, x[1] - 1.2 * L, x[1] + 1.2 * L),
                      col, lw=1.5, ls=(0, (4, 3)), zorder=5)
            if k + 1 < len(RADII):
                nxt = cap_boundary(x, RADII[k + 1], x[0] - L, x[0] + L, x[1] - L, x[1] + L)
                draw_ring(axz, nxt, col, lw=1.8, zorder=6)
            inside = (np.abs(Y - x) <= L).all(1)
            Yi = Y[inside]
            axz.scatter(Yi[:, 0], Yi[:, 1], Yi[:, 2] + 0.002, color=OBS_COLOR, s=48,
                        edgecolors="0.25", linewidths=0.5, depthshade=False, zorder=4)
            axz.scatter(*x, s=85, facecolor="white", edgecolor=col, lw=2.0, depthshade=False, zorder=8)
            axz.set_xlim(x[0] - L, x[0] + L); axz.set_ylim(x[1] - L, x[1] + L); axz.set_zlim(x[2] - L, x[2] + L)
            axz.set_box_aspect((1, 1, 1), zoom=1.7)
            axz.view_init(elev=30, azim=-58)
            # K, Lambda, N_eff printed at the bottom for the caption, not shown in-panel
            axz.set_title(rf"$h = {h}$", fontsize=14, pad=2)
            chain.append((prev_ax, sphere_mesh(x, h), axz, col))
            fig.add_artist(Rectangle((0, 0), 1, 1, transform=axz.transAxes, fill=False, edgecolor=col,
                                     lw=1.2, zorder=20))
            prev_ax = axz

            if k == 0:
                axz.text2D(-0.02, 0.5, name, transform=axz.transAxes, rotation=90, ha="right", va="center",
                           fontsize=14, color=col)

    # ---- connectors, computed after the projections are known
    fig.canvas.draw()
    for src, S, dst, col in chain:
        P = np.column_stack([S[0].ravel(), S[1].ravel(), S[2].ravel()])
        fr = screen_points(src, P)
        if src is ax0:
            pairs = [(fr[np.argmax(fr[:, 0])], (0, 0.5))]
        else:
            pairs = [(fr[np.argmax(fr[:, 1])], (0, 1)), (fr[np.argmin(fr[:, 1])], (0, 0))]
        for pt, corner in pairs:
            fig.add_artist(ConnectionPatch(xyA=tuple(pt), coordsA=fig.transFigure, xyB=corner,
                                           coordsB=dst.transAxes, color=col, lw=1.0, alpha=0.8, zorder=-1))
    fig.savefig(fname + ".png", dpi=200)
    fig.savefig(fname + ".pdf")
    plt.close(fig)


_UG, _VG = np.meshgrid(np.linspace(0.02, 0.98, 900), np.linspace(0.02, 0.98, 900))
_GAM = gamma(np.column_stack([_UG.ravel(), _VG.ravel()]))
_G = np.exp(-((_UG - CTR[0]) ** 2 + (_VG - CTR[1]) ** 2) / (2 * SD ** 2))
_RHO_UV = ((N_CL / N) * _G / _G.mean() + (N_UN / N)) / 0.96 ** 2      # density w.r.t. du dv on the square


def Lambda_exact(x, h):
    """Λ_{N,h}(x) = N P(B(x,h)) computed by quadrature over the (u,v) square."""
    inside = ((_GAM - x) ** 2).sum(1) <= h * h
    return N * (_RHO_UV.ravel() * inside).mean() * 0.96 ** 2


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    zoom_chain(os.path.join(OUT_DIR, "fig2"))

    print()
    print("Figure 2 caption numbers")
    print(f"  N = {N} observations, m = 2, d = 3")
    for name, uv in QUERIES.items():
        x = gamma(uv)[0]
        for h in RADII:
            q = responsibilities(x, h)
            K = int((((Y - x) ** 2).sum(1) <= h * h).sum())
            print(f"    {name:14s} h={h:<6} K_Nh={K:4d}  Lambda_Nh={Lambda_exact(x, h):8.1f}  "
                  f"N_eff={1 / (q ** 2).sum():7.1f}  max q={q.max():.3f}")
    print()
    print(f"wrote -> {os.path.normpath(OUT_DIR)}/fig2.{{pdf,png}}")
