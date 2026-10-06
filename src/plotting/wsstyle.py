"""Shared plotting style and layout helpers for the paper's figures (Okabe-Ito palette, one figure legend, no
per-panel text)."""
import matplotlib as mpl
from matplotlib.lines import Line2D

OKABE_ITO = {
    "blue":     "#0072B2",
    "vermilion": "#D55E00",
    "green":    "#009E73",
    "orange":   "#E69F00",
    "purple":   "#CC79A7",
    "skyblue":  "#56B4E9",
    "yellow":   "#F0E442",
    "black":    "#000000",
}

C = {
    "unmemorized": OKABE_ITO["blue"],
    "memorized":   OKABE_ITO["vermilion"],
    "atoms":       "#8A8A8A",
    "manifold":    "#BBBBBB",
    "boundary":    "#000000",
    "field":       "#7B5EA7",
    "accent":      OKABE_ITO["green"],
    "muted":       "#888888",
}

MARK = {
    "sample_s": 46,
    "sample_alpha": 0.9,
    "sample_edge": 0.7,
    "atom_s": 95,
    "atom_lw": 1.8,
    "atom_alpha": 1.0,
    "curve_lw": 3.0,
    "line_lw": 3.5,
}


def use_style(font_size=16, serif=True):
    """Apply the house rcParams; titles and ticks derive from font_size."""
    family = "serif" if serif else "sans-serif"
    mpl.rcParams.update({
        "font.family": family,
        # STIX mathtext matches Times.
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 3,
        "ytick.labelsize": font_size - 3,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": MARK["line_lw"],
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # TrueType: text stays text in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def grid_on(ax):
    """Faint grid on major and minor ticks."""
    ax.grid(True, which="both", alpha=mpl.rcParams["grid.alpha"])
    ax.set_axisbelow(True)


def strip_axes(ax):
    """Remove ticks, tick labels and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def header_rule(fig, x_extent, y, color="#BBBBBB", lw=1.0):
    """Horizontal hairline under a block header, in figure coordinates."""
    line = Line2D(x_extent, [y, y], transform=fig.transFigure,
                  color=color, lw=lw, zorder=-1, clip_on=False)
    fig.add_artist(line)
    return line


def separator(fig, x, y0, y1, color="#D5D5D5", lw=1.0):
    """Vertical rule in a gutter, in figure coordinates."""
    line = Line2D([x, x], [y0, y1], transform=fig.transFigure,
                  color=color, lw=lw, zorder=-1, clip_on=False)
    fig.add_artist(line)
    return line


def gutter_positions(block_rects):
    """Midpoint x of each gap between consecutive blocks (n_blocks - 1 values)."""
    return [0.5 * (block_rects[i][1] + block_rects[i + 1][0])
            for i in range(len(block_rects) - 1)]


def blend_on_white(color, alpha):
    """Opaque colour that `color` at `alpha` renders as on white (legend keys for translucent bands)."""
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(color)
    return to_hex(tuple(alpha * ch + (1.0 - alpha) * 1.0 for ch in (r, g, b)))


def alpha_ramp_cmap(color, alpha_max=0.42, name="alpha_ramp"):
    """Single-hue colormap ramping alpha from 0 to alpha_max, so a field under a scatter only inks its high end."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    stops = np.linspace(0.0, 1.0, 256)
    rgba = np.column_stack([np.tile(to_rgb(color), (len(stops), 1)), stops * alpha_max])
    return LinearSegmentedColormap.from_list(name, rgba)


def legend_handles(entries):
    """Proxy artists for entries [(label, kind, color[, linestyle])], kind in "point", "cross", "line", "patch"."""
    handles = []
    for entry in entries:
        label, kind, color = entry[:3]
        ls = entry[3] if len(entry) > 3 else "-"
        if kind == "patch":
            from matplotlib.patches import Patch
            h = Patch(facecolor=color, edgecolor="#B4B4B4", linewidth=0.8, label=label)
            handles.append(h)
            continue
        if kind == "point":
            h = Line2D([], [], marker="o", linestyle="none", color=color,
                       markersize=11, markeredgewidth=0, label=label)
        elif kind == "cross":
            h = Line2D([], [], marker="x", linestyle="none", color=color,
                       markersize=12, markeredgewidth=MARK["atom_lw"], label=label)
        else:
            h = Line2D([], [], color=color, lw=MARK["line_lw"], linestyle=ls,
                       label=label)
        handles.append(h)
    return handles


def figure_legend(fig, entries, ncol=None, y=-0.02, loc="upper center"):
    """One legend for the figure, centered below the grid by default."""
    handles = legend_handles(entries)
    return fig.legend(handles=handles, loc=loc, ncol=ncol or len(handles),
                      bbox_to_anchor=(0.5, y), frameon=False,
                      handletextpad=0.4, columnspacing=1.8)


def match_limits(axes, xlim=None, ylim=None, pad=0.06):
    """Shared limits over all panels; missing xlim/ylim default to the padded union of current limits."""
    flat = axes.flat if hasattr(axes, "flat") else axes
    flat = list(flat)
    if xlim is None:
        lo = min(ax.get_xlim()[0] for ax in flat)
        hi = max(ax.get_xlim()[1] for ax in flat)
        m = pad * (hi - lo)
        xlim = (lo - m, hi + m)
    if ylim is None:
        lo = min(ax.get_ylim()[0] for ax in flat)
        hi = max(ax.get_ylim()[1] for ax in flat)
        m = pad * (hi - lo)
        ylim = (lo - m, hi + m)
    for ax in flat:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    return xlim, ylim


ALL_FIGURES = __import__("pathlib").Path(__file__).resolve().parents[2] / "all_figures"


def save(fig, name, out_dir=None, formats=("pdf",)):
    """Write the figure to all_figures/ (uploaded to Overleaf); returns the paths written, relative to the repo root."""
    import os
    if out_dir is None:
        out_dir = ALL_FIGURES
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, format=fmt)
        paths.append(os.path.relpath(p, ALL_FIGURES.parent))
    return paths


def bbox_union(axes_list):
    """Union of axes positions in figure coordinates, as (x0, y0, x1, y1)."""
    boxes = [ax.get_position() for ax in axes_list]
    return (min(b.x0 for b in boxes), min(b.y0 for b in boxes),
            max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def block_header(fig, axes_list, label, x_extent=None, dy=0.052, size_delta=1):
    """Title centered over a block of panels; returns the y it wrote at."""
    x0, _, x1, y1 = bbox_union(axes_list)
    if x_extent is not None:
        x0, x1 = x_extent
    y = y1 + dy
    fig.text(0.5 * (x0 + x1), y, label, ha="center", va="bottom",
             fontsize=mpl.rcParams["font.size"] + size_delta)
    return y


def place_grid(fig, n_rows, block_sizes, left=0.045, right=0.012, top=0.845,
               bottom=0.140, block_gap=0.048, panel_gap=0.013, row_gap=0.040,
               equal_panel_width=True, equal_block_width=True):
    """Lay out panels in blocks side by side; block_sizes: panel columns per block, e.g. [1, 2, 2].

    equal_panel_width keeps panels the same width (sparser blocks centered), so panels stay comparable. Returns (axes
    [n_rows][n_panels], blocks [per-block [n_rows][k]], block_rects [(x0, x1)] for block_frame's x_extent)."""
    n_blocks = len(block_sizes)
    avail_w = 1.0 - left - right
    max_k = max(block_sizes)

    if equal_block_width:
        block_w = (avail_w - block_gap * (n_blocks - 1)) / n_blocks
        uniform_pw = (block_w - panel_gap * (max_k - 1)) / max_k
        block_widths = [block_w] * n_blocks
    else:
        inner = sum(max(k - 1, 0) for k in block_sizes)
        uniform_pw = ((avail_w - block_gap * (n_blocks - 1) - panel_gap * inner)
                      / sum(block_sizes))
        block_widths = [k * uniform_pw + panel_gap * (k - 1) for k in block_sizes]

    panel_h = (top - bottom - row_gap * (n_rows - 1)) / n_rows

    axes = [[] for _ in range(n_rows)]
    blocks = []
    block_rects = []
    bx0 = left
    for b, k in enumerate(block_sizes):
        block_w = block_widths[b]
        block_rects.append((bx0, bx0 + block_w))
        pw = uniform_pw if equal_panel_width else (block_w - panel_gap * (k - 1)) / k
        span = k * pw + panel_gap * (k - 1)
        x_start = bx0 + 0.5 * (block_w - span)
        block_axes = [[] for _ in range(n_rows)]
        for j in range(k):
            x = x_start + j * (pw + panel_gap)
            for i in range(n_rows):
                y = top - panel_h - i * (panel_h + row_gap)
                ax = fig.add_axes([x, y, pw, panel_h])
                axes[i].append(ax)
                block_axes[i].append(ax)
        blocks.append(block_axes)
        bx0 += block_w + block_gap
    return axes, blocks, block_rects


def flatten(nested):
    """[[ax, ax], [ax, ax]] -> [ax, ax, ax, ax]."""
    return [ax for row in nested for ax in row]


METHOD_LABELS = {
    "unguided": "Unguided",
    "fr":       "FR",
    "bayesian_fr": "Bayesian FR",
    "random":   "Random perturbations",
    "amg":      "AMG",
    "bm":       "Broken Memories",
}
METHOD_COLORS = {
    "unguided": OKABE_ITO["black"],
    "fr":       OKABE_ITO["blue"],
    "bayesian_fr": OKABE_ITO["green"],
    "random":   OKABE_ITO["skyblue"],
    "amg":      OKABE_ITO["orange"],
    "bm":       OKABE_ITO["purple"],
}


# Figure widths in inches: FULL_WIDTH at text width, HALF_WIDTH for one of two side by side.
FONT_SIZE, TITLE_SIZE = 13, 14
FULL_WIDTH, HALF_WIDTH = 16.0, 7.8


def paper_style():
    """use_style at the paper's sizes: 13 pt text, 14 pt titles."""
    use_style(font_size=FONT_SIZE)
    mpl.rcParams.update({"axes.titlesize": TITLE_SIZE, "figure.titlesize": TITLE_SIZE})


LARGE_FONT_SIZE = 20


def large_style():
    """paper_style with 20 pt text and 18 pt ticks and legends, for figures shown shrunk in the paper."""
    paper_style()
    s = LARGE_FONT_SIZE
    mpl.rcParams.update({"font.size": s, "axes.titlesize": s, "axes.labelsize": s, "figure.titlesize": s,
                         "xtick.labelsize": s - 2, "ytick.labelsize": s - 2, "legend.fontsize": s - 2})
