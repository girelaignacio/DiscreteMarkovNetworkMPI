#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faceted versions of the country-comparison figures (main-text Fig. 2 and Fig. 3),
matching the aesthetic of the regional facets (plot_regional_facets.py).

Fig. 2  -> Congo (DR) and Zimbabwe        -> images/country_facets_fig2.png
Fig. 3  -> Mali and Guinea-Bissau         -> images/country_facets_fig3.png

Each panel shows the estimated deprivation-linkage graph for one country, with the
multidimensional poverty status Z as a central outcome node. Unlike the regional
figure (where edge width encodes adjacency frequency), here the edge thickness is
FIXED, because each panel is a single country: solid edges are pairwise conditional
log-OR between indicators, dashed spokes are the conditional log-OR between each
indicator and Z. Edge COLOUR encodes the conditional log-OR (RdBu, blue +, red -),
on a SINGLE shared scale across all four countries (one colour bar per figure).

Run:  python code/plot_country_facets.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# NB: plot_networks_logOR.py runs its plotting loop at import time, so we inline
# the (small) shared metadata and helpers here instead of importing it.
INDICATORS = ["d_nutr", "d_cm", "d_educ", "d_satt",
              "d_ckfl", "d_sani", "d_wtr", "d_elct", "d_hsg", "d_asst"]
LABELS = {"d_nutr": "NU", "d_cm": "CM", "d_educ": "YS", "d_satt": "SA",
          "d_ckfl": "CF", "d_sani": "SN", "d_wtr": "DW", "d_elct": "EC",
          "d_hsg": "HO", "d_asst": "AS"}
NODE_COLORS = {"d_nutr": "#962b21", "d_cm": "#652525", "d_educ": "#c6a9ab",
               "d_satt": "#a68580", "d_ckfl": "#afc4d1", "d_sani": "#7d9eb6",
               "d_wtr": "#5e8199", "d_elct": "#3f6781", "d_hsg": "#174d68",
               "d_asst": "#00384f"}
OUTCOME_LABEL = "Z"
_dimensions_indicators = {"hl": ["d_cm", "d_nutr"], "ed": ["d_satt", "d_educ"],
                          "ls": ["d_elct", "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"]}


def _circular_pos(nodes, r=1.0, start_angle=None):
    n = len(nodes)
    if start_angle is None:
        start_angle = np.pi / 2
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
    return {nd: np.array([r * np.cos(a), r * np.sin(a)]) for nd, a in zip(nodes, angles)}


def _indic_weights(mpi_ind):
    iw = {}
    for key, vals in mpi_ind.items():
        w = 1 / len(mpi_ind)
        for v in vals:
            iw[v] = w / len(vals)
    return iw


def _mpi_poor(df):
    iw = _indic_weights(_dimensions_indicators)
    score = df[list(iw.keys())].copy()
    for c in score.columns:
        score[c] *= iw[c]
    score = score.sum(axis=1)
    return (np.where(score >= 1 / 3, score, 0) > 0).astype(int)

PROC_DIR   = os.path.join(REPO, "processed_data")
STABLE_DIR = os.path.join(REPO, "results_stable2")
LOGOR_DIR  = os.path.join(REPO, "results_logOR")
IMG_DIR    = os.path.join(REPO, "images")

# Figures: (output suffix, [(file, display name), ...])
FIGURES = {
    "fig2": [("cog_mics14-15", "Congo (DR)"), ("zwe_mics19", "Zimbabwe")],
    "fig3": [("mli_dhs18", "Mali"), ("gnb_mics18-19", "Guinea-Bissau")],
}

# --- aesthetics (shared with the regional facets) ---
PANEL_SIZE = 7.6        # bigger individual panels
NODE_R = 0.165
Z_R = 0.20
LABEL_FS = 16
TITLE_FS = 19
AXIS_LIM = 1.28         # tighter framing -> larger network, less inter-panel space
LW_EDGE = 7.0           # FIXED width for indicator-indicator edges
LW_SPOKE = 4.5          # FIXED width for indicator-Z spokes
DPI = 200


def _load_country(filename):
    """Return (adj, logOR_matrix, logOR_Y, col_names) for one country."""
    data_path = os.path.join(PROC_DIR, filename)
    adj = np.loadtxt(os.path.join(STABLE_DIR, f"{filename}_mpi_poor_conserv.txt")).astype(int)
    lor_mat = os.path.join(LOGOR_DIR, f"{filename}_logOR_matrix.txt")
    lor_y = os.path.join(LOGOR_DIR, f"{filename}_logOR_Y.txt")
    df = pd.read_csv(data_path, index_col=0).dropna().astype(int)
    if os.path.exists(lor_mat) and os.path.exists(lor_y):
        logOR_matrix = np.loadtxt(lor_mat)
        logOR_Y = np.loadtxt(lor_y).ravel()
    else:
        from discrete_gm_nonpos import discrete_graphical_model
        X = df.to_numpy().astype(int)
        Y = _mpi_poor(df).reshape(-1, 1)
        dgm = discrete_graphical_model()
        logOR_matrix, logOR_Y = dgm.compute_interaction_logOR(
            X, Y, ne=adj, smoothing=1.0, symmetrize=True)
    return adj, logOR_matrix, logOR_Y, df.columns.tolist()


def _country_edge_vals(adj, logOR_matrix, logOR_Y, col_names):
    """All displayed log-OR values for a country (for shared colour range)."""
    ci = {c: i for i, c in enumerate(col_names)}
    vals = []
    p = len(INDICATORS)
    for i in range(p):
        for j in range(i + 1, p):
            ni, nj = INDICATORS[i], INDICATORS[j]
            if ni in ci and nj in ci:
                ii, jj = ci[ni], ci[nj]
                if adj[ii, jj] == 1 and not np.isnan(logOR_matrix[ii, jj]):
                    vals.append(logOR_matrix[ii, jj])
    for i in range(p):
        ni = INDICATORS[i]
        if ni in ci and not np.isnan(logOR_Y[ci[ni]]):
            vals.append(logOR_Y[ci[ni]])
    return vals


def _draw_country(ax, adj, logOR_matrix, logOR_Y, col_names, norm, cmap, title):
    ci = {c: i for i, c in enumerate(col_names)}
    p = len(INDICATORS)
    pos = _circular_pos(INDICATORS, r=1.0)
    zc = np.array([0.0, 0.0])

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-AXIS_LIM, AXIS_LIM); ax.set_ylim(-AXIS_LIM, AXIS_LIM)

    # spokes to Z (dashed, fixed width) — drawn first so they sit beneath edges
    for i in range(p):
        ni = INDICATORS[i]
        if ni in ci and not np.isnan(logOR_Y[ci[ni]]):
            lor = logOR_Y[ci[ni]]
            ax.plot([pos[ni][0], zc[0]], [pos[ni][1], zc[1]],
                    color=cmap(norm(lor)), linewidth=LW_SPOKE, linestyle=(0, (4, 3)),
                    alpha=0.85, zorder=1)

    # indicator-indicator edges (solid, fixed width)
    for i in range(p):
        for j in range(i + 1, p):
            ni, nj = INDICATORS[i], INDICATORS[j]
            if ni in ci and nj in ci:
                ii, jj = ci[ni], ci[nj]
                if adj[ii, jj] == 1 and not np.isnan(logOR_matrix[ii, jj]):
                    lor = logOR_matrix[ii, jj]
                    ax.plot([pos[ni][0], pos[nj][0]], [pos[ni][1], pos[nj][1]],
                            color=cmap(norm(lor)), linewidth=LW_EDGE,
                            solid_capstyle="round", alpha=0.92, zorder=2)

    # nodes
    for ind in INDICATORS:
        x, y = pos[ind]
        ax.add_patch(plt.Circle((x, y), NODE_R, facecolor=NODE_COLORS[ind],
                                zorder=3, linewidth=1.6, edgecolor="white"))
        ax.text(x, y, LABELS[ind], ha="center", va="center",
                fontsize=LABEL_FS, fontweight="bold", color="white", zorder=4)
    # outcome node Z
    ax.add_patch(plt.Circle(tuple(zc), Z_R, facecolor="white",
                            zorder=3, linewidth=2.0, edgecolor="#444444"))
    ax.text(zc[0], zc[1], OUTCOME_LABEL, ha="center", va="center",
            fontsize=LABEL_FS, fontweight="bold", color="#333333", zorder=4)

    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)


def main():
    # Pass 1: one shared colour range across ALL four countries (both figures)
    loaded, all_vals = {}, []
    for fig_key, members in FIGURES.items():
        for filename, name in members:
            adj, lm, ly, cols = _load_country(filename)
            loaded[filename] = (adj, lm, ly, cols)
            all_vals += _country_edge_vals(adj, lm, ly, cols)
    vmax = max(abs(min(all_vals)), abs(max(all_vals)), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu
    letters = ["a", "b", "c", "d"]

    # Pass 2: render each figure (2 panels) with a shared colour bar (placed in a
    # dedicated top band so it never overlaps the panels)
    for fig_key, members in FIGURES.items():
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_SIZE, PANEL_SIZE * 1.10))
        axes = np.atleast_1d(axes).ravel()
        for k, (filename, name) in enumerate(members):
            adj, lm, ly, cols = loaded[filename]
            _draw_country(axes[k], adj, lm, ly, cols, norm, cmap,
                          f"({letters[k]}) {name}")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.0)
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cax = fig.add_axes([0.34, 0.945, 0.32, 0.022])   # x, y, w, h (figure fraction)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Conditional log-OR", fontsize=16, labelpad=8)
        cbar.ax.tick_params(labelsize=12)
        out = os.path.join(IMG_DIR, f"country_facets_{fig_key}.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")
    print(f"Shared log-OR vmax across all four countries = {vmax:.2f}")


if __name__ == "__main__":
    main()
