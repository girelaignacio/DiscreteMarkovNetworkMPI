#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROTOTYPE (for review): country-comparison facets (main-text Fig. 2 and Fig. 3)
in the SAME style as the regional node-size figure: the effect of each indicator
on the poverty status Z is encoded in the NODE SIZE (the indicator-Z conditional
log-OR, which is positive for all indicators), instead of a central Z node with spokes.

Edges: fixed thickness, colour = pairwise conditional log-OR between indicators,
on a single shared scale across all four countries. Node radius increases with the
(positive) indicator-Z conditional log-OR (shared node-size scale across the four countries).
Node fill colour still encodes the MPI dimension.

Outputs (do NOT touch the manuscript):
    images/country_facets_fig2_nodeZ.png   (Congo, Zimbabwe)
    images/country_facets_fig3_nodeZ.png   (Mali, Guinea-Bissau)

Run:  python code/plot_country_facets_nodeZ.py
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

from plot_country_facets import (  # noqa: E402  (guarded by __main__, safe to import)
    _load_country, INDICATORS, LABELS, NODE_COLORS, _circular_pos, FIGURES, IMG_DIR,
)

# aesthetics
PANEL_SIZE = 4.0
LABEL_FS = 13
TITLE_FS = 14
AXIS_LIM = 1.24
LW_EDGE = 7.5                  # FIXED edge width (single country per panel)
DPI = 200
SHOW_LEGENDS = True     # legends embedded on the figure
R_MIN, R_MAX = 0.13, 0.245    # node radius range from indicator-Z log-OR (min raised so labels fit)
LEG_VALUES = [1.0, 2.0, 5.0]


def _radius(absval, amax):
    if not np.isfinite(absval):
        return R_MIN
    return R_MIN + (R_MAX - R_MIN) * min(absval / amax, 1.0)


def _draw(ax, adj, logOR_matrix, logOR_Y, col_names, amax, norm, cmap, title):
    ci = {c: i for i, c in enumerate(col_names)}
    p = len(INDICATORS)
    pos = _circular_pos(INDICATORS, r=1.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-AXIS_LIM, AXIS_LIM); ax.set_ylim(-AXIS_LIM, AXIS_LIM)

    for i in range(p):
        for j in range(i + 1, p):
            ni, nj = INDICATORS[i], INDICATORS[j]
            if ni in ci and nj in ci:
                ii, jj = ci[ni], ci[nj]
                if adj[ii, jj] == 1 and not np.isnan(logOR_matrix[ii, jj]):
                    lor = logOR_matrix[ii, jj]
                    ax.plot([pos[ni][0], pos[nj][0]], [pos[ni][1], pos[nj][1]],
                            color=cmap(norm(lor)), linewidth=LW_EDGE,
                            solid_capstyle="round", alpha=0.92, zorder=1)

    for ind in INDICATORS:
        x, y = pos[ind]
        aY = abs(logOR_Y[ci[ind]]) if ind in ci else np.nan
        r = _radius(aY, amax)
        ax.add_patch(plt.Circle((x, y), r, facecolor="white",
                                zorder=2, linewidth=1.6, edgecolor="black"))
        ax.text(x, y, LABELS[ind], ha="center", va="center",
                fontsize=LABEL_FS, fontweight="bold", color="black", zorder=3)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)


def _add_size_legend(fig, axpos, amax, label_fs=16, val_fs=14):
    """Compact node-size key: circles whose RELATIVE areas follow the node radius
    mapping (a conventional size key, not a 1:1 physical replica)."""
    lax = fig.add_axes(axpos)
    lax.set_aspect("equal"); lax.axis("off")
    lax.set_xlim(0, 3.6); lax.set_ylim(-0.65, 0.85)
    lax.text(0.0, 0.78, "Conditional log-OR wrt $Z$ (node size)",
             fontsize=label_fs, va="bottom", ha="left")
    scale = 1.7
    x = 0.45
    for v in LEG_VALUES:
        r = _radius(v, amax) * scale
        lax.add_patch(plt.Circle((x, 0.0), r, facecolor="white",
                                 edgecolor="black", linewidth=1.4))
        lax.text(x, -0.58, f"{v:g}", ha="center", va="top", fontsize=val_fs)
        x += 1.25


def _add_size_legend_vertical(fig, axpos, amax, label_fs=10.5, val_fs=10.5):
    """Node-size key with the circles stacked vertically (title wrapped on top)."""
    lax = fig.add_axes(axpos)
    lax.set_aspect("equal"); lax.axis("off")
    lax.set_xlim(0, 2.0); lax.set_ylim(-0.3, 4.0)
    lax.text(0.0, 3.95, "Conditional\nlog-OR wrt $Z$\n(node size)",
             fontsize=label_fs, va="top", ha="left", linespacing=1.1)
    scale = 1.7
    y = 2.45            # circles start just below the (3-line) title -> small gap
    for v in LEG_VALUES:
        r = _radius(v, amax) * scale
        lax.add_patch(plt.Circle((0.55, y), r, facecolor="white",
                                 edgecolor="black", linewidth=1.4))
        lax.text(1.2, y, f"{v:g}", va="center", ha="left", fontsize=val_fs)
        y -= 0.82


def main():
    # shared scales across ALL four countries (both figures)
    loaded, edge_abs, y_abs = {}, [], []
    for members in FIGURES.values():
        for filename, name in members:
            adj, lm, ly, cols = _load_country(filename)
            loaded[filename] = (adj, lm, ly, cols)
            ci = {c: i for i, c in enumerate(cols)}
            p = len(INDICATORS)
            for i in range(p):
                for j in range(i + 1, p):
                    ni, nj = INDICATORS[i], INDICATORS[j]
                    if ni in ci and nj in ci and adj[ci[ni], ci[nj]] == 1 \
                            and not np.isnan(lm[ci[ni], ci[nj]]):
                        edge_abs.append(abs(lm[ci[ni], ci[nj]]))
            for ind in INDICATORS:
                if ind in ci and np.isfinite(ly[ci[ind]]):
                    y_abs.append(abs(ly[ci[ind]]))
    vmax = max(max(edge_abs) if edge_abs else 1e-6, 1e-6)
    amax = max(float(np.percentile(y_abs, 95)) if y_abs else 1e-6, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    # all four countries in a single 2x2 figure (Fig. 2): (a)-(d)
    members = [c for grp in FIGURES.values() for c in grp]
    letters = ["a", "b", "c", "d"]
    fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_SIZE, 2 * PANEL_SIZE * 1.02))
    axes = axes.ravel()
    for k, (filename, name) in enumerate(members):
        adj, lm, ly, cols = loaded[filename]
        _draw(axes[k], adj, lm, ly, cols, amax, norm, cmap, f"({letters[k]}) {name}")
    fig.subplots_adjust(left=0.01, right=0.80, top=0.95, bottom=0.03,
                        wspace=0.04, hspace=0.15)

    # shared legends on the RIGHT: node-size key (top) + short colour bar (centre);
    # both titles wrapped on top, numbers on the right.
    _add_size_legend_vertical(fig, [0.81, 0.55, 0.18, 0.40], amax)
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.885, 0.24, 0.022, 0.20])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_title("Conditional\nlog-OR\n(edges)", fontsize=10.5, linespacing=1.1)

    out = os.path.join(IMG_DIR, "country_facets_combined_nodeZ.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  [scales] edge_vmax={vmax:.4f}  node_amax={amax:.4f}")


if __name__ == "__main__":
    main()
