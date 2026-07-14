#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global deprivation-linkage graph (main-text Fig. 3a) in the SAME style as the
regional facets: edge colour = mean conditional lOR (RdBu), edge width
proportional to the share of countries with the association, node size
increasing with the (positive) indicator-Z conditional lOR pooled over all 63
countries. Embedded legends: node-size key, edge-width key, lOR colour bar.

Output: images/global_graph_logOR_nodeZ.png
Run:    python code/plot_global_nodeZ.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_global_graph import (  # noqa: E402
    aggregate, _edge_list, _circular_pos,
    INDICATORS, CSV_IX, LABELS, NODE_COLORS, B_ALPHA, IMG_DIR,
)
from plot_regional_facets_nodeZ import (  # noqa: E402  (guarded by __main__)
    _mean_logOR_Y, _radius, EW_MIN, EW_MAX, AXIS_LIM, LEG_VALUES,
)

LABEL_FS = 16
DPI = 200
WIDTH_LEG_VALUES = [0.25, 0.50, 1.00]    # share of countries shown in the width key


def _draw(ax, freq, mean_logOR, mean_lY, amax, norm, cmap):
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


def _add_size_legend(fig, axpos, amax, label_fs=10.5, val_fs=10):
    lax = fig.add_axes(axpos)
    lax.set_aspect("equal"); lax.axis("off")
    lax.set_xlim(0, 3.6); lax.set_ylim(-0.65, 1.45)
    lax.text(0.0, 0.70, "Conditional log-OR wrt $Z$\n(node size)",
             fontsize=label_fs, va="bottom", ha="left", linespacing=1.1)
    scale = 1.7
    x = 0.45
    for v in LEG_VALUES:
        r = _radius(v, amax) * scale
        lax.add_patch(plt.Circle((x, 0.0), r, facecolor="white",
                                 edgecolor="black", linewidth=1.4))
        lax.text(x, -0.58, f"{v:g}", ha="center", va="top", fontsize=val_fs)
        x += 1.25


def _add_width_legend(fig, axpos, label_fs=10.5, val_fs=10):
    """Edge-width key: line thickness = share of countries with the association."""
    lax = fig.add_axes(axpos)
    lax.axis("off")
    lax.set_xlim(0, 3.6); lax.set_ylim(-0.65, 1.45)
    lax.text(0.0, 0.70, "Share of countries\n(edge width)",
             fontsize=label_fs, va="bottom", ha="left", linespacing=1.1)
    x = 0.15
    for v in WIDTH_LEG_VALUES:
        lw = EW_MIN + (EW_MAX - EW_MIN) * v
        lax.plot([x, x + 0.75], [0.05, 0.05], color="#9aa6ad",
                 linewidth=lw, solid_capstyle="round")
        lax.text(x + 0.375, -0.58, f"{int(v*100)}%",
                 ha="center", va="top", fontsize=val_fs)
        x += 1.20


def main():
    freq, mean_logOR, auc, n, sign, ecs = aggregate(None)
    mean_lY = _mean_logOR_Y(None)
    edge_abs = [abs(mean_logOR[CSV_IX[a], CSV_IX[b]]) for a, b, _ in _edge_list(freq)
                if np.isfinite(mean_logOR[CSV_IX[a], CSV_IX[b]])]
    y_abs = [abs(v) for v in mean_lY if np.isfinite(v)]
    vmax = max(float(np.percentile(edge_abs, 95)) if edge_abs else 1e-6, 1e-6)
    amax = max(float(np.percentile(y_abs, 95)) if y_abs else 1e-6, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    fig = plt.figure(figsize=(6.4, 7.3))
    ax = fig.add_axes([0.02, 0.20, 0.96, 0.78])
    _draw(ax, freq, mean_logOR, mean_lY, amax, norm, cmap)

    # bottom legend strip: node-size key | edge-width key | colour bar
    # (both keys use the same data ratio 3.6/2.1 = box ratio, so they align)
    _add_size_legend(fig, [0.02, 0.0, 0.30, 0.154], amax)
    _add_width_legend(fig, [0.37, 0.0, 0.30, 0.154])
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.73, 0.046, 0.21, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label("Mean conditional\nlog-OR (edges)", fontsize=10.5, labelpad=5)
    cbar.ax.tick_params(labelsize=9)

    out = os.path.join(IMG_DIR, "global_graph_logOR_nodeZ.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}  (n={n}, edge vmax p95={vmax:.2f}, node amax p95={amax:.2f})")


if __name__ == "__main__":
    main()
