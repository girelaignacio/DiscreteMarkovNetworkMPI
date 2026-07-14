#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact heatmaps of adjacency frequencies (lower triangle, Blues) with the
average conditional AUC of each indicator on the diagonal (dark red), in the
style of the original heatmap_*.png figures but sized for the new layout:

  images/heatmap_global_compact.png   -- single global heatmap, legible at
                                         ~half text width (side by side with
                                         the global graph in Fig. 3)
  images/heatmap_regions_facets.png   -- all six world regions as panels
                                         (a)-(f), shared colour bar

Cell annotations: mean over countries, with (+/- sd) underneath.
Run:  python code/plot_heatmaps.py
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
    _build_panels, INDICATORS, CSV_IX, LABELS,
    STABLE_DIR, CONTRIB_DIR, SCENARIO, SYM, IMG_DIR,
)

REGION_ORDER = [
    "Arab States", "South Asia", "Sub-Saharan Africa",
    "Latin America and the Caribbean", "East Asia and the Pacific",
    "Europe and Central Asia",
]
PANEL_LETTERS = ["a", "b", "c", "d", "e", "f"]
DIAG_COLOR = "#8c2d2d"
DPI = 220


def collect(allowed_isos=None):
    """Per-region stacks: adjacency (binary) and AUC per indicator across countries.
    Returns freq_mean, freq_sd (p x p) and auc_mean, auc_sd (p,)."""
    files = sorted(glob.glob(os.path.join(STABLE_DIR, f"*_{SCENARIO}_{SYM}.txt")))
    if allowed_isos is not None:
        files = [f for f in files if os.path.basename(f)[:3] in allowed_isos]
    p = len(CSV_IX)
    adjs, aucs = [], []
    for f in files:
        country = os.path.basename(f)[: -len(f"_{SCENARIO}_{SYM}.txt")]
        adjs.append((np.loadtxt(f) > 0).astype(float))
        auc_path = os.path.join(CONTRIB_DIR, f"AUC_{country}_{SCENARIO}_{SYM}.txt")
        row = np.full(p, np.nan)
        if os.path.exists(auc_path):
            with open(auc_path) as fh:
                names = fh.readline().strip().split(",")
                vals = np.array(fh.readline().strip().split(","), dtype=float)
            for k, nm in enumerate(names[:p]):
                if nm in CSV_IX:
                    row[CSV_IX[nm]] = vals[k]
        aucs.append(row)
    A = np.stack(adjs)            # (n, p, p)
    U = np.stack(aucs)            # (n, p)
    freq_mean = A.mean(axis=0)
    freq_sd = A.std(axis=0)
    auc_mean = np.nanmean(U, axis=0)
    auc_sd = np.nanstd(U, axis=0)
    return freq_mean, freq_sd, auc_mean, auc_sd, len(files)


def draw(ax, freq_mean, freq_sd, auc_mean, auc_sd, annot_fs=8.0, tick_fs=9.5,
         show_xticks=True, show_yticks=True):
    """Lower-triangle heatmap in INDICATORS display order; diagonal = AUC."""
    p = len(INDICATORS)
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    ax.set_xlim(0, p); ax.set_ylim(p, 0)
    ax.set_aspect("equal")
    for i in range(p):           # row (display)
        for j in range(i + 1):   # col <= row -> lower triangle incl. diagonal
            ii, jj = CSV_IX[INDICATORS[i]], CSV_IX[INDICATORS[j]]
            if i == j:
                m, s = auc_mean[ii], auc_sd[ii]
                tc = "black"
                # diagonal: white, no box -- just the annotations
            else:
                m, s = freq_mean[ii, jj], freq_sd[ii, jj]
                fc = cmap(norm(m))
                tc = "white" if m > 0.55 else "#222222"
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc,
                                           edgecolor="white", linewidth=1.0))
            if np.isfinite(m):
                ax.text(j + 0.5, i + 0.40, f"{m:.2f}", ha="center", va="center",
                        fontsize=annot_fs, fontweight="bold", color=tc)
                ax.text(j + 0.5, i + 0.76, f"(±{s:.2f})", ha="center",
                        va="center", fontsize=annot_fs * 0.78, color=tc)
    ax.set_xticks(np.arange(p) + 0.5)
    ax.set_yticks(np.arange(p) + 0.5)
    ax.set_xticklabels([LABELS[ind] for ind in INDICATORS] if show_xticks else [],
                       fontsize=tick_fs)
    ax.set_yticklabels([LABELS[ind] for ind in INDICATORS] if show_yticks else [],
                       fontsize=tick_fs)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def make_global(out_name="heatmap_global_compact.png"):
    fm, fs, am, asd, n = collect(None)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    draw(ax, fm, fs, am, asd, annot_fs=8.6, tick_fs=11)
    sm = mcm.ScalarMappable(cmap=plt.cm.Blues,
                            norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Frequencies", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.99, bottom=0.06)
    out = os.path.join(IMG_DIR, out_name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}  (n={n})")


def make_regional(out_name="heatmap_regions_facets.png"):
    panels = {lab: allowed for lab, allowed, _ in _build_panels()}
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.6, 8.6))
    axes = axes.ravel()
    for k, reg in enumerate(REGION_ORDER):
        fm, fs, am, asd, n = collect(panels[reg])
        draw(axes[k], fm, fs, am, asd, annot_fs=6.4, tick_fs=8)
        axes[k].set_title(f"({PANEL_LETTERS[k]}) {reg}", fontsize=12,
                          fontweight="bold", pad=6)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.10,
                        wspace=0.12, hspace=0.16)
    sm = mcm.ScalarMappable(cmap=plt.cm.Blues,
                            norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.35, 0.035, 0.30, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Frequencies", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    out = os.path.join(IMG_DIR, out_name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_global()
    make_regional()
