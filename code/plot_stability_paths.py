#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meinshausen & Buhlmann (2010) Fig. 15 style figures of the stability-selection
SELECTION PROBABILITIES produced by stability_selection_diagnostics.py.

One figure per (country, scenario, symmetrization), two panels:

  LEFT  -- stability path: cumulative selection probability Pi (y, 0..1) for each
           of the 45 possible edges against the regularization lambda (x, log,
           drawn with large lambda on the left so paths rise left->right).  The
           edges that are stable at the selected lambda are highlighted and
           labelled; a dashed vertical line marks lambda*, a dotted line marks the
           stability threshold pi_thr(lambda*), and the shaded band is the
           PFER-accepted lambda window.

  RIGHT -- heatmap of the 10x10 selection-probability matrix at the selected
           lambda, indicators in publication display order, stable edges outlined
           in red (and, optionally, the production results_stable2 edges outlined
           with a black dashed border for cross-comparison).

Run (from anywhere, after running stability_selection_diagnostics.py):
    python code/plot_stability_paths.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Indicator metadata (display order + 2-letter labels + dimension colours),
# kept in sync with code/plot_networks_logOR.py
# ---------------------------------------------------------------------------
INDICATORS = [
    "d_nutr", "d_cm",
    "d_educ", "d_satt",
    "d_ckfl", "d_sani", "d_wtr", "d_elct", "d_hsg", "d_asst",
]
LABELS = {
    "d_nutr": "NU", "d_cm": "CM",
    "d_educ": "YS", "d_satt": "SA",
    "d_ckfl": "CF", "d_sani": "SN",
    "d_wtr":  "DW", "d_elct": "EC",
    "d_hsg":  "HO", "d_asst": "AS",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COUNTRIES = {
    "zwe_mics19":    "Zimbabwe",
    "gnb_mics18-19": "Guinea-Bissau",
    "cog_mics14-15": "Congo (DR)",
    "mli_dhs18":     "Mali",
}
SCENARIOS = ["mpi_poor"]
SYMMETRIZATIONS = ["conserv"]            # "conserv" and/or "nconserv"
MARK_STABLE2 = False                     # overlay results_stable2 edges on the heatmap (off for paper)
SHOW_PER_LAMBDA_OVERLAY = False          # faint non-cumulative Pi = mean(NE,axis=0)
INCLUDE_HEATMAP = False                  # per-country figure = stability path only (no Pi heatmap)
MAKE_FACET = True                        # one combined figure of all stability paths
DPI = 200

IN_DIR      = os.path.join(REPO, "results_stability_paths")
IMG_DIR     = os.path.join(REPO, "images")
STABLE2_DIR = os.path.join(REPO, "results_stable2")

SYM_IX = {"conserv": 0, "nconserv": 1}


def _load(country, scenario):
    path = os.path.join(IN_DIR, f"{country}_{scenario}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _edge_curves(d, sym_ix):
    """Yield (i, j, curve, is_stable) for every upper-triangle edge."""
    selprob = d["selprob"][:, sym_ix, :, :]            # (|c|, p, p)
    key = "conserv" if sym_ix == 0 else "nconserv"
    adj = d[f"final_adj_{key}"]
    p = selprob.shape[1]
    for i in range(p):
        for j in range(i + 1, p):
            yield i, j, selprob[:, i, j], bool(adj[i, j])


def _draw_path(ax, d, sym_ix, names):
    c = d["c_grid"]                                    # descending
    isel = int(d["index_selected"][sym_ix])
    thr = d["pi_thr"][:, sym_ix]
    acc = d["accepted_q"][:, sym_ix]
    NE = d["NE"] if (SHOW_PER_LAMBDA_OVERLAY and d["NE"].size) else None

    stable = [(i, j, curve) for (i, j, curve, st) in _edge_curves(d, sym_ix) if st]
    # non-stable curves: thin grey
    for i, j, curve, st in _edge_curves(d, sym_ix):
        if not st:
            ax.plot(c, curve, color="0.8", lw=0.8, zorder=1)
    # stable curves: distinct colours + end labels
    cmap = plt.get_cmap("tab10")
    legend_handles = []
    for k, (i, j, curve) in enumerate(stable):
        col = cmap(k % 10)
        lab = f"{LABELS[names[i]]}-{LABELS[names[j]]}"
        ax.plot(c, curve, color=col, lw=2.0, zorder=3)
        if NE is not None:
            per_lambda = NE[:, :, sym_ix, i, j].mean(axis=0)
            ax.plot(c, per_lambda, color=col, lw=1.0, ls=":", alpha=0.55, zorder=2)
        # edges are identified by the "stable edges" legend (no end-of-line labels,
        # which otherwise pile up where the curves plateau at Pi=1)
        legend_handles.append(Line2D([0], [0], color=col, lw=2, label=lab))

    # PFER-accepted lambda window (shaded)
    if acc.any():
        cc = c[acc]
        ax.axvspan(cc.min(), cc.max(), color="0.85", alpha=0.35, zorder=0)
    # selected lambda + threshold guides
    if isel >= 0:
        ax.axvline(c[isel], color="k", ls="--", lw=1.0, alpha=0.8, zorder=4)
        ax.axhline(thr[isel], color="crimson", ls=":", lw=1.3, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(c.min(), c.max())
    ax.invert_xaxis()                                  # large lambda on the left
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"regularization $\lambda$  (log; stronger $\to$ left)")
    ax.set_ylabel(r"cumulative selection probability $\Pi$")
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.3)

    # annotation box
    if isel >= 0:
        adj = d["final_adj_conserv"] if sym_ix == 0 else d["final_adj_nconserv"]
        txt = (rf"$\lambda^*$ = {c[isel]:.3g}" + "\n"
               rf"$q$ = {d['Eqhat'][isel, sym_ix]:.2f}" + "\n"
               rf"$\pi_{{thr}}$ = {thr[isel]:.2f}" + "\n"
               rf"edges = {int(adj.sum() // 2)}" + "\n"
               rf"PFER = {float(d['PFER']):.0f}")
        ax.text(0.985, 0.025, txt, transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
                  title="stable edges", title_fontsize=7, framealpha=0.9)


def _draw_heatmap(ax, d, sym_ix, names, country, scenario):
    isel = int(d["index_selected"][sym_ix])
    p = len(names)
    perm = [names.index(ind) for ind in INDICATORS]    # CSV order -> display order
    key = "conserv" if sym_ix == 0 else "nconserv"
    adj = d[f"final_adj_{key}"]

    M = d["selprob"][isel, sym_ix].astype(float)
    Md = M[np.ix_(perm, perm)].copy()
    np.fill_diagonal(Md, np.nan)
    adj_d = adj[np.ix_(perm, perm)]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = ax.imshow(Md, vmin=0, vmax=1, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"selection probability $\Pi$ at $\lambda^*$", fontsize=8)

    disp_labels = [LABELS[ind] for ind in INDICATORS]
    ax.set_xticks(range(p)); ax.set_yticks(range(p))
    ax.set_xticklabels(disp_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(disp_labels, fontsize=8)
    ax.set_xticks(np.arange(-.5, p, 1), minor=True)
    ax.set_yticks(np.arange(-.5, p, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)

    # recomputed stable edges -> red cell border
    for r in range(p):
        for cc in range(p):
            if r != cc and adj_d[r, cc]:
                ax.add_patch(Rectangle((cc - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="red", lw=2.0, zorder=5))
    handles = [Line2D([0], [0], color="red", lw=2, label="stable (reported)")]

    # production results_stable2 edges -> black dashed border
    if MARK_STABLE2:
        ref_path = os.path.join(STABLE2_DIR, f"{country}_{scenario}_{key}.txt")
        if os.path.exists(ref_path):
            ref = (np.loadtxt(ref_path) > 0).astype(int)
            ref_d = ref[np.ix_(perm, perm)]
            for r in range(p):
                for cc in range(p):
                    if r != cc and ref_d[r, cc]:
                        ax.add_patch(Rectangle((cc - 0.42, r - 0.42), 0.84, 0.84,
                                               fill=False, edgecolor="black",
                                               lw=1.2, ls="--", zorder=6))
            handles.append(Line2D([0], [0], color="black", lw=1.2, ls="--",
                                  label="results_stable2"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.08),
              fontsize=7, ncol=len(handles), frameon=False)
    ax.set_title(r"$\Pi$ matrix at $\lambda^*$", fontsize=10)


def make_figure(country, disp, scenario, sym):
    d = _load(country, scenario)
    if d is None:
        print(f"  [skip] no npz for {country}/{scenario}")
        return
    sym_ix = SYM_IX[sym]
    if int(d["index_selected"][sym_ix]) < 0:
        print(f"  [skip] {country}/{scenario}/{sym}: no lambda selected")
        return
    names = list(d["indicator_names"])

    if INCLUDE_HEATMAP:
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1.0]})
        _draw_path(axL, d, sym_ix, names)
        _draw_heatmap(axR, d, sym_ix, names, country, scenario)
    else:                                    # stability path only (no Pi heatmap)
        fig, axL = plt.subplots(1, 1, figsize=(6.5, 5.0))   # ~1.3 ratio (more squared)
        _draw_path(axL, d, sym_ix, names)
    fig.suptitle(f"{disp} — stability path", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"{country}_{scenario}_{sym}_stability.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def make_facet(scenario, sym):
    """One row of stability-path panels across all countries."""
    items = [(c, disp) for c, disp in COUNTRIES.items() if _load(c, scenario) is not None]
    if not items:
        return
    sym_ix = SYM_IX[sym]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, (country, disp) in zip(axes[0], items):
        d = _load(country, scenario)
        if int(d["index_selected"][sym_ix]) < 0:
            ax.set_visible(False); continue
        _draw_path(ax, d, sym_ix, list(d["indicator_names"]))
        ax.set_title(disp, fontsize=11, fontweight="bold")
    fig.suptitle(f"Stability paths — {scenario} — {sym}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(IMG_DIR, f"stability_paths_facet_{scenario}_{sym}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    print("### Plotting stability paths ###")
    for scenario in SCENARIOS:
        for sym in SYMMETRIZATIONS:
            for country, disp in COUNTRIES.items():
                make_figure(country, disp, scenario, sym)
            if MAKE_FACET:
                make_facet(scenario, sym)


if __name__ == "__main__":
    main()
