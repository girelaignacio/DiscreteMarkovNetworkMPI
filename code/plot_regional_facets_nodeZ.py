#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROTOTYPE (for review): regional facets (main-text Fig. 4 style) where each
indicator's NODE SIZE encodes its effect on the poverty status Z, i.e. the
region-average conditional log-OR between the indicator and Z (positive for all indicators).
No central node / no spokes -> avoids the clutter of the with-Z version.

Indicator-indicator edges: solid, width proportional to the relative frequency of
adjacency occurrence in the region, colour = mean conditional log-OR (as in Fig. 4).
Node radius: increasing in the (positive) region-average indicator-Z log-OR (shared scale
across all panels). Node fill colour still encodes the MPI dimension.

Output (does NOT touch the manuscript): images/global_graph_logOR_facets_nodeZ.png

Run:  python code/plot_regional_facets_nodeZ.py
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from plot_global_graph import (  # noqa: E402
    aggregate, _build_panels, _edge_list, _circular_pos,
    INDICATORS, CSV_IX, CSV_ORDER, LABELS, NODE_COLORS, B_ALPHA,
    STABLE_DIR, SCENARIO, SYM, IMG_DIR,
)

REGION_ORDER = [
    "Arab States", "South Asia", "Sub-Saharan Africa",
    "Latin America and the Caribbean", "East Asia and the Pacific",
    "Europe and Central Asia",
]
PANEL_LETTERS = ["a", "b", "c", "d", "e", "f"]
LOGOR_DIR = os.path.join(REPO, "results_logOR")

# aesthetics (consistent with plot_regional_facets.py)
PANEL_SIZE = 4.0
LABEL_FS = 13
TITLE_FS = 14
EW_MIN, EW_MAX = 1.0, 12.0
AXIS_LIM = 1.24
NCOLS = 3
DPI = 200
SHOW_LEGENDS = True     # legends embedded on the figure

# node radius mapping from indicator-Z log-OR (positive for all indicators)
R_MIN, R_MAX = 0.13, 0.245    # node radius range from indicator-Z log-OR (min raised so labels fit)
LEG_VALUES = [1.0, 2.0, 5.0]      # reference log-OR values in the size legend


def _mean_logOR_Y(allowed_isos):
    files = sorted(glob.glob(os.path.join(STABLE_DIR, f"*_{SCENARIO}_{SYM}.txt")))
    if allowed_isos is not None:
        files = [f for f in files if os.path.basename(f)[:3] in allowed_isos]
    p = len(CSV_ORDER)
    s = np.zeros(p); c = np.zeros(p)
    for f in files:
        country = os.path.basename(f)[: -len(f"_{SCENARIO}_{SYM}.txt")]
        yp = os.path.join(LOGOR_DIR, f"{country}_logOR_Y.txt")
        if not os.path.exists(yp):
            continue
        ly = np.loadtxt(yp).ravel()
        if ly.shape[0] != p:
            continue
        m = ~np.isnan(ly)
        s[m] += ly[m]; c[m] += 1
    out = np.full(p, np.nan)
    out[c > 0] = s[c > 0] / c[c > 0]
    return out


def _radius(absval, amax):
    if not np.isfinite(absval):
        return R_MIN
    return R_MIN + (R_MAX - R_MIN) * min(absval / amax, 1.0)


def _draw(ax, freq, mean_logOR, mean_lY, amax, norm, cmap, title):
    pos = _circular_pos(INDICATORS, r=1.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-AXIS_LIM, AXIS_LIM); ax.set_ylim(-AXIS_LIM, AXIS_LIM)

    for a, b, f in _edge_list(freq):
        lor = mean_logOR[CSV_IX[a], CSV_IX[b]]
        if np.isnan(lor):
            continue
        t = min(f / 1.0, 1.0)
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=cmap(norm(lor)), linewidth=EW_MIN + (EW_MAX - EW_MIN) * t,
                alpha=B_ALPHA, solid_capstyle="round", zorder=1)

    for ind in INDICATORS:
        x, y = pos[ind]
        r = _radius(abs(mean_lY[CSV_IX[ind]]), amax)
        ax.add_patch(plt.Circle((x, y), r, facecolor="white",
                                zorder=2, linewidth=1.6, edgecolor="black"))
        ax.text(x, y, LABELS[ind], ha="center", va="center",
                fontsize=LABEL_FS, fontweight="bold", color="black", zorder=3)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=2)


def _add_size_legend(fig, axpos, amax, label_fs=13, val_fs=12):
    """Compact node-size key: circles whose RELATIVE areas follow the same radius
    mapping as the nodes (a conventional size key, not a 1:1 physical replica)."""
    lax = fig.add_axes(axpos)
    lax.set_aspect("equal"); lax.axis("off")
    lax.set_xlim(0, 3.6); lax.set_ylim(-0.65, 1.45)
    lax.text(0.0, 0.70, "Conditional log-OR wrt $Z$\n(node size)",
             fontsize=label_fs, va="bottom", ha="left", linespacing=1.1)
    scale = 1.7   # enlarge the key circles for visibility (relative sizes preserved)
    x = 0.45
    for v in LEG_VALUES:
        r = _radius(v, amax) * scale
        lax.add_patch(plt.Circle((x, 0.0), r, facecolor="white",
                                 edgecolor="black", linewidth=1.4))
        lax.text(x, -0.58, f"{v:g}", ha="center", va="top", fontsize=val_fs)
        x += 1.25


WIDTH_LEG_VALUES = [0.25, 0.50, 1.00]    # share of countries shown in the width key


def _add_width_legend(fig, axpos, label_fs=13, val_fs=12):
    """Edge-width key: line thickness = share of countries with the association."""
    lax = fig.add_axes(axpos)
    lax.axis("off")
    lax.set_xlim(0, 3.6); lax.set_ylim(-0.65, 1.45)
    lax.text(0.0, 0.70, "Share of countries\n(edge width)",
             fontsize=label_fs, va="bottom", ha="left", linespacing=1.1)
    x = 0.15
    for v in WIDTH_LEG_VALUES:
        lw = EW_MIN + (EW_MAX - EW_MIN) * v
        lax.plot([x, x + 0.80], [0.05, 0.05], color="#9aa6ad",
                 linewidth=lw, solid_capstyle="round")
        lax.text(x + 0.40, -0.58, f"{int(v*100)}%", ha="center", va="top",
                 fontsize=val_fs)
        x += 1.25


def main():
    panels = {lab: allowed for lab, allowed, _ in _build_panels()}
    aggs, edge_abs, y_abs = {}, [], []
    for reg in REGION_ORDER:
        allowed = panels[reg]
        freq, mean_logOR, auc, n, sign, ecs = aggregate(allowed)
        mean_lY = _mean_logOR_Y(allowed)
        aggs[reg] = (freq, mean_logOR, mean_lY, n)
        for a, b, _ in _edge_list(freq):
            v = mean_logOR[CSV_IX[a], CSV_IX[b]]
            if np.isfinite(v):
                edge_abs.append(abs(v))
        y_abs += [abs(v) for v in mean_lY if np.isfinite(v)]
    vmax = max(float(np.percentile(edge_abs, 95)) if edge_abs else 1e-6, 1e-6)
    amax = max(float(np.percentile(y_abs, 95)) if y_abs else 1e-6, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    nrows = int(np.ceil(len(REGION_ORDER) / NCOLS))
    FIGW, FIGH = NCOLS * PANEL_SIZE, nrows * PANEL_SIZE
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(FIGW, FIGH))
    axes = np.atleast_1d(axes).ravel()
    for k, reg in enumerate(REGION_ORDER):
        freq, mean_logOR, mean_lY, n = aggs[reg]
        _draw(axes[k], freq, mean_logOR, mean_lY, amax, norm, cmap,
              f"({PANEL_LETTERS[k]}) {reg}")
    for ax in axes[len(REGION_ORDER):]:
        ax.axis("off")
    top = 0.95
    bottom = 0.14 if SHOW_LEGENDS else 0.01
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom,
                        wspace=0.0, hspace=0.13)

    if SHOW_LEGENDS:
        # legends embedded at the BOTTOM: node-size key (left) + edge-width key
        # (centre) + colour bar (right); all labels placed above their reference.
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cax = fig.add_axes([0.68, 0.045, 0.22, 0.014])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.set_label("Mean conditional log-OR (edges)", fontsize=14, labelpad=6)
        cbar.ax.tick_params(labelsize=12)
        # both key boxes share the data ratio 3.6/2.1 so their scales align
        _add_size_legend(fig, [0.035, 0.0, 0.171, 0.15], amax)
        _add_width_legend(fig, [0.27, 0.0, 0.171, 0.15])
    # report the scales so the LaTeX/TikZ legend can match them exactly
    print(f"  [scales] edge_vmax={vmax:.4f}  node_amax={amax:.4f}")

    out = os.path.join(IMG_DIR, "global_graph_logOR_facets_nodeZ.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}  (edge vmax p95={vmax:.2f}; node |logOR_Y| p95={amax:.2f})")


if __name__ == "__main__":
    main()
