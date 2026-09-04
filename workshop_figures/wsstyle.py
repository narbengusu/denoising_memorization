"""Shared plotting style for the workshop-paper figures.

Base style follows the DR-Control `plotter.ipynb` reference: large type, thick
lines, faint `which="both"` grid, PDF output. Extended here with a
colorblind-safe categorical palette (Okabe-Ito) and the panel/legend helpers
the reference notebook does not have, because our figures are grids rather
than single axes.

House rules these helpers enforce, and the reason each exists:

- **No per-panel text.** A conference figure is read at a glance, and every
  hyperparameter printed inside an axes competes with the data for that
  glance. Panel identity goes in short column titles on the FIRST ROW ONLY
  (`column_titles`) and row labels on the LEFT COLUMN ONLY (`row_labels`);
  numeric settings go in the LaTeX caption, never in the figure.
- **One legend for the whole figure**, placed outside every axes
  (`figure_legend`), because repeating an identical legend in each panel of a
  grid wastes the plotting area it covers.
- **Shared limits across a grid** (`match_limits`), so panels sitting side by
  side are actually comparable rather than each auto-scaled to its own data.
- **Marks sized to survive print.** `MARK` carries larger `s`/`lw` than
  matplotlib's defaults; at two-column width a default `s=32` scatter point is
  roughly a pixel on paper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito, the standard colorblind-safe categorical set. Chosen over
# matplotlib's tab10 because tab10's blue/orange pair is the one that fails
# most often under deuteranopia, and these figures lean on a two-class
# (memorized / not memorized) color contrast as their primary signal.
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

# Semantic roles. Plot code should reference these, never a raw hex string, so
# that "memorized" is the same red in every figure of the paper.
C = {
    "unmemorized": OKABE_ITO["blue"],
    "memorized":   OKABE_ITO["vermilion"],
    "atoms":       "#8A8A8A",
    "manifold":    "#BBBBBB",
    "boundary":    "#000000",
    # Background field shading (Fisher-Rao energy). Deliberately far in hue from
    # BOTH sample colours -- a ground that sits near either the blue or the
    # orange steals contrast from the marks it is meant to sit behind.
    "field":       "#7B5EA7",
    "accent":      OKABE_ITO["green"],
    "muted":       "#888888",
}

# Mark specs. Kept here rather than at call sites so point size stays
# consistent across figures that are placed next to each other in the paper.
# Draw order matters more than usual in these figures: memorized samples sit
# ON TOP of the training atoms by definition, so if atoms are drawn last they
# cover exactly the points the figure exists to show. Atoms go UNDERNEATH
# (Z["atoms"]), memorized samples go on top of everything, and memorized
# samples carry a thin light edge so a cluster of them stays readable against
# the black cross beneath it.
Z = {"manifold": 0, "atoms": 1, "unmemorized": 2, "memorized": 3}

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
    """Apply the house rcParams.

    font_size is the BASE size; titles and ticks are derived from it. Pass a
    smaller value for wide multi-column grids (each panel is physically small,
    so 16pt type would swamp it) and the default for one- or two-panel
    figures. Everything downstream reads from rcParams, so a figure never
    hardcodes a fontsize.

    serif=True matches the Computer Modern body text of a LaTeX paper, so
    figure type and caption type look like the same document. Set False if the
    paper template uses a sans body font.
    """
    family = "serif" if serif else "sans-serif"
    mpl.rcParams.update({
        "font.family": family,
        # Times New Roman first. mathtext uses "stix" rather than "cm" because
        # STIX is metrically designed to sit alongside Times -- Computer Modern
        # next to Times makes every "$N = 10$" visibly lighter and narrower than
        # the roman text beside it.
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
        # Keep text as text in the PDF so the paper's own font machinery and
        # any reviewer's text search still work on the figure.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def grid_on(ax):
    """The reference notebook's grid: faint, and on both major and minor
    ticks so log axes stay readable."""
    ax.grid(True, which="both", alpha=mpl.rcParams["grid.alpha"])
    ax.set_axisbelow(True)


def strip_axes(ax):
    """Remove ticks, tick labels, and all four spines.

    For the scatter/image panels, the axis NUMBERS carry no information a
    reader of this paper needs -- the claim is about where points sit relative
    to the training atoms and the curve, not about absolute coordinates -- and
    five columns of tick labels is exactly the visual clutter the figure is
    trying to avoid.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def column_titles(axes, titles, pad=10):
    """Short titles on the TOP ROW only. axes: the 2-D array from subplots."""
    row0 = axes[0] if axes.ndim > 1 else axes
    for ax, title in zip(row0, titles):
        ax.set_title(title, pad=pad)


def row_labels(axes, labels, pad=12):
    """Labels on the LEFT COLUMN only, written as a y-label so they rotate and
    align with the panel rather than floating in figure coordinates."""
    col0 = axes[:, 0] if axes.ndim > 1 else [axes[0]]
    for ax, label in zip(col0, labels):
        ax.set_ylabel(label, labelpad=pad)


def group_headers(fig, axes, groups, y=1.0, pad=0.022):
    """Second-level headers spanning several columns, e.g. one "Ambient" label
    over two adjacent panels.

    groups: list of (label, first_col, last_col) with inclusive column
    indices. The label is centered over the span in FIGURE coordinates, read
    off the actual axes positions, so it stays centered no matter how
    `tight_layout`/`constrained_layout` ends up sizing the panels -- call this
    AFTER the layout is finalized.
    """
    row0 = axes[0] if axes.ndim > 1 else axes
    for label, lo, hi in groups:
        x0 = row0[lo].get_position().x0
        x1 = row0[hi].get_position().x1
        fig.text(0.5 * (x0 + x1), y + pad, label,
                 ha="center", va="bottom",
                 fontsize=mpl.rcParams["font.size"] + 1)


def header_rule(fig, x_extent, y, color="#BBBBBB", lw=1.0):
    """A hairline spanning a block, drawn under its header.

    Replaces the filled card as the group separator. A rule plus whitespace
    carries the same grouping information as a shaded box while adding almost
    no non-data ink, which is what the surrounding figures in a proceedings
    volume will look like."""
    from matplotlib.lines import Line2D
    line = Line2D(x_extent, [y, y], transform=fig.transFigure,
                  color=color, lw=lw, zorder=-1, clip_on=False)
    fig.add_artist(line)
    return line


def alpha_ramp_cmap(color="#9A9A9A", alpha_max=0.62, name="alpha_ramp"):
    """A single-hue colormap that fades in via ALPHA rather than lightness.

    For a field laid under a scatter, a normal sequential map tints the whole
    panel and drags every mark's contrast down with it. Ramping alpha instead
    leaves low values fully transparent, so only the high end -- the part the
    figure is actually pointing at -- puts ink on the page, and the samples
    and curve keep their contrast everywhere else."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    r, g, b = to_rgb(color)
    stops = np.linspace(0.0, 1.0, 256)
    rgba = np.column_stack([np.full_like(stops, r), np.full_like(stops, g),
                            np.full_like(stops, b), stops * alpha_max])
    return LinearSegmentedColormap.from_list(name, rgba)


def separator(fig, x, y0, y1, color="#D5D5D5", lw=1.0):
    """A vertical light rule in a gutter, in figure coordinates.

    Pairs with `header_rule` to separate groups using lines and whitespace
    instead of a filled background: the horizontals sit under the group
    headers, the verticals run down the gutters between groups, and together
    they read as a light rule grid rather than as boxes."""
    from matplotlib.lines import Line2D
    line = Line2D([x, x], [y0, y1], transform=fig.transFigure,
                  color=color, lw=lw, zorder=-1, clip_on=False)
    fig.add_artist(line)
    return line


def gutter_positions(block_rects):
    """Midpoint x of each gap BETWEEN consecutive blocks -- where `separator`
    lines go. Returns one x per gap, so n_blocks - 1 of them."""
    return [0.5 * (block_rects[i][1] + block_rects[i + 1][0])
            for i in range(len(block_rects) - 1)]


def add_strip(fig, ax, frac=0.20, gap=0.04):
    """Carve a short marginal strip off the bottom of `ax`, returning the new
    strip axes (and shrinking `ax` to make room).

    Used for a distribution that shares the panel's x-axis -- here, where
    samples fall along the curve. Placed by hand rather than via a gridspec so
    it composes with `place_grid`'s manual layout.

    frac and gap are fractions of the ORIGINAL axes height.
    """
    pos = ax.get_position()
    strip_h = pos.height * frac
    gap_h = pos.height * gap
    ax.set_position([pos.x0, pos.y0 + strip_h + gap_h,
                     pos.width, pos.height - strip_h - gap_h])
    strip = fig.add_axes([pos.x0, pos.y0, pos.width, strip_h])
    return strip


def project_to_curve(X, curve_xy, curve_param):
    """Nearest-point projection of samples onto a densely sampled curve,
    returning each sample's curve PARAMETER.

    Turns a 2-D scatter into the 1-D "where along the manifold did this land"
    coordinate that the coverage strip histograms. Brute force against the
    dense scan -- fine at these sizes and exact, unlike an analytic inverse
    that would have to be rederived per manifold."""
    import torch
    idx = torch.cdist(X, curve_xy).argmin(dim=1)
    return curve_param[idx]


def blend_on_white(color, alpha):
    """The opaque colour that `color` at `alpha` actually renders as on a white page.

    Use it to build a legend key for a translucent band: passing the band's own colour
    at full opacity makes the key several times darker than the thing it stands for,
    which is the same mismatch a mid-colormap sample creates for a continuously
    coloured mark."""
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(color)
    return to_hex(tuple(alpha * ch + (1.0 - alpha) * 1.0 for ch in (r, g, b)))


def legend_handles(entries):
    """entries: list of (label, kind, color) or (label, kind, color, linestyle),
    where kind is "point", "cross", "line", or "patch" (a filled swatch, for a
    shaded band such as a guidance window). Returns proxy artists sized by
    MARK, so the legend keys are legible even when the plotted marks are small
    and semi-transparent."""
    handles = []
    for entry in entries:
        label, kind, color = entry[:3]
        ls = entry[3] if len(entry) > 3 else "-"
        if kind == "patch":
            from matplotlib.patches import Patch
            # A faint edge: patch keys are usually pale shaded bands, and a swatch
            # that matches such a band exactly is nearly invisible without one.
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
    """One legend for the entire figure, anchored OUTSIDE every axes.

    Default position is centered below the grid. Every panel in these figures
    plots the same three or four categories, so a per-panel legend would be
    the same key repeated N times, each one covering data.
    """
    handles = legend_handles(entries)
    return fig.legend(handles=handles, loc=loc, ncol=ncol or len(handles),
                      bbox_to_anchor=(0.5, y), frameon=False,
                      handletextpad=0.4, columnspacing=1.8)


def match_limits(axes, xlim=None, ylim=None, pad=0.06):
    """Force one shared data range over every panel in a grid.

    If xlim/ylim are not given they are computed as the union of what each
    axes currently holds, then padded. Without this, `sharex`/`sharey` alone
    still leaves panels auto-scaled differently whenever they are created in
    separate calls, and a reader comparing two panels side by side would be
    comparing different zoom levels -- which is exactly the comparison
    Figure 1 asks them to make.
    """
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


def save(fig, name, out_dir=None, formats=("pdf", "png")):
    """Write the figure to workshop_figures/out/ in every requested format.

    PDF is the one that goes into LaTeX (vector, so it stays sharp at any
    column width); PNG exists only for quick viewing in a notebook or a slide
    deck. Returns the list of paths written.
    """
    import os
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


def bbox_union(axes_list):
    """Union of several axes' positions, in FIGURE coordinates.

    Read off live positions rather than the gridspec, so it stays correct for
    axes placed by hand with `add_axes` as well as ones from `subplots`."""
    boxes = [ax.get_position() for ax in axes_list]
    return (min(b.x0 for b in boxes), min(b.y0 for b in boxes),
            max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def block_frame(fig, axes_list, x_extent=None, pad_x=0.024, pad_y=0.058,
                color="#C9C9C9", lw=1.0, radius=0.016, facecolor="#F5F5F5"):
    """Draw one rounded border around a GROUP of panels, making it read as a
    distinct sub-figure.

    A grid of bare, spine-less panels has no visual boundary between one
    conceptual group and the next, so five panels in a row read as five
    equally-related things. Boxing each group states the actual structure --
    baseline, ambient, tangent-projected -- before the reader parses any
    label. Drawn in figure coordinates and added to `fig.patches`, so it sits
    behind the axes and is unaffected by anything inside them.

    The default is a filled CARD (light face, soft border, generous padding)
    rather than a hairline rectangle. A thin outline around tightly-packed
    panels is easy to miss; a filled ground separates the groups at a glance,
    which is the whole point of drawing them. Pass facecolor="none" for an
    outline-only frame.

    x_extent=(x0, x1) overrides the horizontal span, which is what
    `place_grid`'s `block_rects` supplies. Without it the frame is drawn
    around the union of the axes it contains -- so a block holding ONE
    centered panel would get a frame half the width of a two-panel block,
    defeating the equal-width layout. Pass the block's own rect to make all
    the frames the same width regardless of how many panels sit inside.

    Call AFTER panel positions are final.
    """
    from matplotlib.patches import FancyBboxPatch
    x0, y0, x1, y1 = bbox_union(axes_list)
    if x_extent is not None:
        x0, x1 = x_extent
    patch = FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y), (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=facecolor, edgecolor=color,
        # zorder must be strictly BELOW the axes' default of 0: figure-level
        # children are drawn in zorder order and ties break on insertion, so a
        # filled card added after the axes would paint straight over them.
        linewidth=lw, zorder=-1, clip_on=False, mutation_aspect=1.0)
    fig.patches.append(patch)
    return patch


def block_header(fig, axes_list, label, x_extent=None, dy=0.052, size_delta=1):
    """Group title centered over a block of panels, in figure coordinates.

    Returns the y it wrote at, so a caller can align several headers or place a
    frame clear of them."""
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
    """Lay out panels as several equal-width BLOCKS side by side.

    block_sizes: how many panel columns each block holds, e.g. [1, 2, 2] for
    unguided / ambient / tangent-projected.

    Every block gets the SAME share of the figure width regardless of how many
    panels it holds, so the three groups carry equal visual weight and the
    baseline is not squeezed into a fifth of the page.

    equal_panel_width=True (default) then keeps every PANEL the same width as
    the panels in the busiest block and centers the sparser blocks' panels
    inside their share. This is the point of the flag: letting a 1-panel block
    stretch to fill a 2-panel-wide share would draw that panel at twice the
    horizontal scale of the ones it is meant to be compared against, so the
    same sinusoid would appear stretched in the baseline and compressed in the
    guided panels. Padding with whitespace instead keeps all panels directly
    comparable. Set False to let each block's panels fill their share.

    Returns (axes, blocks, block_rects): `axes` is a [n_rows][n_panels]
    nested list in left-to-right order, `blocks` is a list of per-block
    [n_rows][k] lists (what block_frame / block_header take), and
    `block_rects` is a list of (x0, x1) figure-coordinate spans -- one per
    block, all the same width -- to hand to block_frame's `x_extent` so the
    frames come out equal even where a block holds fewer panels.
    """
    n_blocks = len(block_sizes)
    avail_w = 1.0 - left - right
    max_k = max(block_sizes)

    if equal_block_width:
        block_w = (avail_w - block_gap * (n_blocks - 1)) / n_blocks
        uniform_pw = (block_w - panel_gap * (max_k - 1)) / max_k
        block_widths = [block_w] * n_blocks
    else:
        # Every PANEL the same width, every BLOCK only as wide as it needs to
        # be. Equal block widths only earn their keep when the blocks are drawn
        # as visible boxes that would look lopsided otherwise; separated by
        # whitespace alone, a one-panel block padded out to two-panel width
        # just reads as an unexplained hole.
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
        x_start = bx0 + 0.5 * (block_w - span)          # centered within the block
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
    """[[ax, ax], [ax, ax]] -> [ax, ax, ax, ax]; for handing a block to
    block_frame/block_header/match_limits."""
    return [ax for row in nested for ax in row]


# Display names for the guidance methods, in ONE place. The paper calls the
# `bayesian_fr.py` energy "Bayesian FR" with no version suffix -- `v1`/`v2`
# are internal module names and must not reach a figure. Rename here and every
# figure updates.
METHOD_LABELS = {
    "unguided": "Unguided",
    "basic":    "FR",
    "bayesian": "Bayesian FR",
}
