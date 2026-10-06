"""Shared setup for the surface figures: N points on K = {(u,v,f(u,v))} in R^3, dense cluster + uniform background."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from plotting import wsstyle

OUT_DIR = str(wsstyle.ALL_FIGURES)
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
import contourpy

wsstyle.large_style()

SEED = 4
rng = np.random.default_rng(SEED)


def f(u, v):
    return (0.30 * np.exp(-((u - 0.28) ** 2 + (v - 0.62) ** 2) / 0.06)
            - 0.22 * np.exp(-((u - 0.72) ** 2 + (v - 0.35) ** 2) / 0.08))


def gamma(uv):
    uv = np.atleast_2d(uv)
    return np.column_stack([uv[:, 0], uv[:, 1], f(uv[:, 0], uv[:, 1])])


N_CL, N_UN, CTR, SD = 300, 120, np.array([0.30, 0.58]), 0.11


def sample(rng):
    pts = []
    while len(pts) < N_CL:
        p = rng.normal(CTR, SD, size=(N_CL, 2))
        p = p[(p > 0.02).all(1) & (p < 0.98).all(1)]
        pts += list(p)
    return np.vstack([np.array(pts[:N_CL]), rng.uniform(0.02, 0.98, size=(N_UN, 2))])


UV = sample(rng)
Y = gamma(UV)
N = len(Y)
QUERIES = {"dense region": (0.30, 0.58), "sparse region": (0.59, 0.22)}
RADII = [0.25, 0.10, 0.04]
ROW_COLORS = {"dense region": "#1F7A8C", "sparse region": "#7B4EA3"}

OBS_COLOR = "#E87C3A"
LS = LightSource(azdeg=300, altdeg=50)
SURF_COLOR = "#E9EAEE"


def responsibilities(x, h):
    d2 = ((Y - x) ** 2).sum(1)
    lg = -d2 / (2 * h * h)
    lg -= lg.max()
    q = np.exp(lg)
    return q / q.sum()


def surface_patch(ax, u0, u1, v0, v1, n=90, zorder=1, alpha=0.96):
    U, V = np.meshgrid(np.linspace(u0, u1, n), np.linspace(v0, v1, n))
    ax.plot_surface(U, V, f(U, V), color=SURF_COLOR, shade=True, lightsource=LS,
                    linewidth=0, antialiased=True, alpha=alpha, zorder=zorder)


def sphere_mesh(x, h):
    ph, th = np.mgrid[0:np.pi:31j, 0:2 * np.pi:61j]
    return (x[0] + h * np.sin(ph) * np.cos(th), x[1] + h * np.sin(ph) * np.sin(th), x[2] + h * np.cos(ph))


def draw_ball(ax, x, h, col, face_alpha=0.10, zorder=2):
    S = sphere_mesh(x, h)
    ax.plot_surface(*S, color=col, alpha=face_alpha, linewidth=0, shade=False, zorder=zorder)
    ax.plot_wireframe(*S, rstride=5, cstride=10, color=col, lw=0.45, alpha=0.30, zorder=zorder)


def cap_boundary(x, h, u0, u1, v0, v1, n=500):
    """Boundary of B(x,h) ∩ K."""
    U, V = np.meshgrid(np.linspace(u0, u1, n), np.linspace(v0, v1, n))
    G = (U - x[0]) ** 2 + (V - x[1]) ** 2 + (f(U, V) - x[2]) ** 2 - h * h
    lines = contourpy.contour_generator(U, V, G).lines(0.0)
    return [np.column_stack([l[:, 0], l[:, 1], f(l[:, 0], l[:, 1]) + 0.002]) for l in lines]


def draw_ring(ax, curves, col, lw, ls="-", zorder=5):
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=col, lw=lw, ls=ls, zorder=zorder)


def clean3d(ax):
    ax.set_axis_off()
    ax.computed_zorder = False
    ax.patch.set_visible(False)


def clean(ax):
    ax.patch.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def screen_points(ax, P):
    xp, yp, _ = proj3d.proj_transform(P[:, 0], P[:, 1], P[:, 2], ax.get_proj())
    disp = ax.transData.transform(np.column_stack([xp, yp]))
    return ax.figure.transFigure.inverted().transform(disp)
