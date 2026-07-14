#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate the GLOBAL findings network (paper Fig. 5a, "global_graph.png") in the
same visual style as the per-country plots in plot_networks_logOR.py.

The global network aggregates the stable graphs of all 63 countries:
    freq[i, j] = mean over countries of the binary adjacency
                 (results_stable2/<country>_mpi_poor_conserv.txt)
This is exactly the off-diagonal of heatmap_None.png (verified: NU-YS 0.76,
AS-EC 0.60, NU-CM 0.03, ...).

Three figures per panel, all with UNIFORM-size nodes restyled to the per-country
dimension palette (circular layout, white 2-letter labels):

  global_graph[_<region>].png             -- edges GRAY, width/opacity prop. to freq.
  global_graph_logOR[_<region>].png       -- edges coloured by mean conditional log-OR
                                             (blue = +, red = -, RdBu), width prop. freq.
  global_graph_logOR_split[_<region>].png -- one edge per sign (countries that vote +
                                             vs -), minority countries annotated.

A panel is the global set (all 63 countries -> global_graph*.png) plus one set per
world region (-> *_<region>.png, e.g. _Sub-Saharan_Africa, matching heatmap_<region>).
With COMPARABLE_SCALES, edge width = absolute frequency and the log-OR colour range is
shared across all panels so regions can be read against each other.

Run (from anywhere):
    python code/plot_global_graph.py
"""

import os
import sys
import glob
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

# ---------------------------------------------------------------------------
# Indicator metadata + style (copied from plot_networks_logOR.py to stay in sync;
# importing that module would execute its top-level country loop)
# ---------------------------------------------------------------------------
INDICATORS = [                       # circular display order
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
NODE_COLORS = {
    "d_nutr": "#962b21", "d_cm": "#652525",
    "d_educ": "#c6a9ab", "d_satt": "#a68580",
    "d_ckfl": "#afc4d1", "d_sani": "#7d9eb6",
    "d_wtr":  "#5e8199", "d_elct": "#3f6781",
    "d_hsg":  "#174d68", "d_asst": "#00384f",
}
# Column order of processed_data CSVs == row/col order of results_stable2 matrices
CSV_ORDER = ["d_cm", "d_nutr", "d_satt", "d_educ", "d_elct",
             "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"]
CSV_IX = {name: i for i, name in enumerate(CSV_ORDER)}

_dimensions_indicators = {
    "hl": ["d_cm", "d_nutr"],
    "ed": ["d_satt", "d_educ"],
    "ls": ["d_elct", "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"],
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO = "mpi_poor"
SYM      = "conserv"
NODE_RADIUS = 0.13                   # uniform node size (user choice)
W_MIN, W_MAX = 0.4, 11.0             # edge width range (mapped from frequency)
A_MIN, A_MAX = 0.12, 0.95            # gray-edge opacity range (Plot A)
B_ALPHA = 0.92                       # constant-ish opacity for log-OR edges (Plot B)
DPI = 150
COMPUTE_LOGOR = True                 # needed for the log-OR variants
NODE_SIZE_BY_AUC = False             # original Fig 5a sized nodes by mean AUC; off per request
COMPARABLE_SCALES = True             # absolute width (=freq) + shared log-OR colour range
PER_REGION = True                    # also emit one figure set per world region
SHOW_TITLE = False                   # figure titles (off for paper; captions added externally)
SHOW_OUTLIER_NOTES = False           # sign-mixed minority-country note box (off for paper)

PROC_DIR    = os.path.join(REPO, "processed_data")
STABLE_DIR  = os.path.join(REPO, "results_stable2")
CONTRIB_DIR = os.path.join(REPO, "results_contributions")
IMG_DIR     = os.path.join(REPO, "images")
LOOKUP_XLSX = os.path.join(REPO, "utils", "Table 1 National Results MPI 2024.xlsx")


# ---------------------------------------------------------------------------
# Helpers (copied from plot_networks_logOR.py)
# ---------------------------------------------------------------------------
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


def _circular_pos(nodes, r=1.0, start_angle=None):
    """Evenly spaced circular positions starting at top (pi/2)."""
    n = len(nodes)
    if start_angle is None:
        start_angle = np.pi / 2
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
    return {nd: np.array([r * np.cos(a), r * np.sin(a)]) for nd, a in zip(nodes, angles)}


def load_region_map():
    """ISO(lowercase 3-letter) -> world region, from the OPHI MPI table."""
    iso2region = {}
    try:
        raw = pd.read_excel(LOOKUP_XLSX, header=None, skiprows=9)
    except Exception as e:
        print(f"  [warn] could not read region lookup ({e}); per-region disabled")
        return iso2region
    for _, r in raw.iterrows():
        iso = str(r[1]).strip()
        if len(iso) == 3 and iso.isalpha():
            iso2region[iso.lower()] = str(r[3]).strip()
    return iso2region


# ---------------------------------------------------------------------------
# Aggregate across countries
# ---------------------------------------------------------------------------
def aggregate(allowed_isos=None):
    """Aggregate stable graphs across countries. If allowed_isos is given, restrict
    to countries whose 3-letter ISO (filename prefix) is in that set."""
    files = sorted(glob.glob(os.path.join(STABLE_DIR, f"*_{SCENARIO}_{SYM}.txt")))
    if allowed_isos is not None:
        files = [f for f in files if os.path.basename(f)[:3] in allowed_isos]
    if not files:
        raise SystemExit(f"No adjacency files in {STABLE_DIR} matching *_{SCENARIO}_{SYM}.txt")

    p = len(CSV_ORDER)
    adj_sum = np.zeros((p, p))
    lo_sum  = np.zeros((p, p))
    lo_cnt  = np.zeros((p, p))
    pos_cnt = np.zeros((p, p)); pos_sum = np.zeros((p, p))   # per-sign across countries
    neg_cnt = np.zeros((p, p)); neg_sum = np.zeros((p, p))
    # per-edge country membership (i<j in CSV order): {"pos":[(iso,country,logOR)],...}
    edge_country_signs = {(i, j): {"pos": [], "neg": [], "zero": []}
                          for i in range(p) for j in range(i + 1, p)}
    auc_sum = np.zeros(p)
    auc_cnt = np.zeros(p)
    n = 0

    dgm = None
    if COMPUTE_LOGOR:
        from discrete_gm_nonpos import discrete_graphical_model
        dgm = discrete_graphical_model()

    for f in files:
        country = os.path.basename(f)[: -len(f"_{SCENARIO}_{SYM}.txt")]
        adj = (np.loadtxt(f) > 0).astype(int)
        adj_sum += adj
        n += 1

        if COMPUTE_LOGOR or NODE_SIZE_BY_AUC:
            data_path = os.path.join(PROC_DIR, country)
            if not os.path.exists(data_path):
                print(f"  [warn] no processed_data for {country}; skipping its logOR")
            elif COMPUTE_LOGOR:
                df = pd.read_csv(data_path, index_col=0).dropna().astype(int)
                X = df[CSV_ORDER].to_numpy().astype(int)
                Y = _mpi_poor(df).reshape(-1, 1).astype(int)
                logOR_matrix, _ = dgm.compute_interaction_logOR(
                    X, Y, ne=adj, smoothing=1.0, symmetrize=True)
                m = ~np.isnan(logOR_matrix)
                lo_sum[m] += logOR_matrix[m]
                lo_cnt[m] += 1
                pm = m & (logOR_matrix > 0); nm = m & (logOR_matrix < 0)
                pos_cnt[pm] += 1; pos_sum[pm] += logOR_matrix[pm]
                neg_cnt[nm] += 1; neg_sum[nm] += logOR_matrix[nm]
                iso = country[:3]
                for i in range(p):
                    for j in range(i + 1, p):
                        if m[i, j]:
                            v = float(logOR_matrix[i, j])
                            grp = "pos" if v > 0 else ("neg" if v < 0 else "zero")
                            edge_country_signs[(i, j)][grp].append((iso, country, v))

        if NODE_SIZE_BY_AUC:
            auc_path = os.path.join(CONTRIB_DIR, f"AUC_{country}_{SCENARIO}_{SYM}.txt")
            if os.path.exists(auc_path):
                with open(auc_path) as fh:
                    names = fh.readline().strip().split(",")
                    vals = np.array(fh.readline().strip().split(","), dtype=float)
                for k, nm in enumerate(names[:len(CSV_ORDER)]):
                    if nm in CSV_IX:
                        auc_sum[CSV_IX[nm]] += vals[k]
                        auc_cnt[CSV_IX[nm]] += 1

    freq = adj_sum / n
    mean_logOR = np.full((p, p), np.nan)
    nz = lo_cnt > 0
    mean_logOR[nz] = lo_sum[nz] / lo_cnt[nz]
    auc = np.where(auc_cnt > 0, auc_sum / np.maximum(auc_cnt, 1), np.nan)
    mean_pos = np.full((p, p), np.nan); mean_neg = np.full((p, p), np.nan)
    mean_pos[pos_cnt > 0] = pos_sum[pos_cnt > 0] / pos_cnt[pos_cnt > 0]
    mean_neg[neg_cnt > 0] = neg_sum[neg_cnt > 0] / neg_cnt[neg_cnt > 0]
    sign = dict(pos_cnt=pos_cnt, neg_cnt=neg_cnt, mean_pos=mean_pos, mean_neg=mean_neg)
    return freq, mean_logOR, auc, n, sign, edge_country_signs


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _node_radii(auc):
    if not NODE_SIZE_BY_AUC or np.all(np.isnan(auc)):
        return {ind: NODE_RADIUS for ind in INDICATORS}
    a = np.clip(auc, 0.5, None)
    lo, hi = np.nanmin(a), np.nanmax(a)
    rng = max(hi - lo, 1e-9)
    return {ind: 0.09 + 0.10 * (np.clip(auc[CSV_IX[ind]], 0.5, None) - lo) / rng
            for ind in INDICATORS}


def _draw_nodes(ax, pos, radii):
    for ind in INDICATORS:
        x, y = pos[ind]
        ax.add_patch(plt.Circle((x, y), radii[ind], facecolor=NODE_COLORS[ind],
                                 zorder=2, linewidth=1.5, edgecolor="white"))
        ax.text(x, y, LABELS[ind], ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=3)


def _edge_list(freq):
    """Yield (a, b, f) for INDICATOR pairs with frequency > 0."""
    for ai in range(len(INDICATORS)):
        for bi in range(ai + 1, len(INDICATORS)):
            ia, ib = CSV_IX[INDICATORS[ai]], CSV_IX[INDICATORS[bi]]
            f = freq[ia, ib]
            if f > 0:
                yield INDICATORS[ai], INDICATORS[bi], f


def _add_title(ax, title):
    if SHOW_TITLE and title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=8)


def plot_gray(freq, auc, out_path, title=None):
    pos = _circular_pos(INDICATORS, r=1.0)
    radii = _node_radii(auc)
    fmax = 1.0 if COMPARABLE_SCALES else max((f for *_, f in _edge_list(freq)), default=1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)

    for a, b, f in _edge_list(freq):
        t = min(f / fmax, 1.0)
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="0.35", linewidth=W_MIN + (W_MAX - W_MIN) * t,
                alpha=A_MIN + (A_MAX - A_MIN) * t,
                solid_capstyle="round", zorder=1)
    _draw_nodes(ax, pos, radii)
    _add_title(ax, title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_logOR(freq, mean_logOR, auc, out_path, title=None, vmax=None):
    pos = _circular_pos(INDICATORS, r=1.0)
    radii = _node_radii(auc)
    fmax = 1.0 if COMPARABLE_SCALES else max((f for *_, f in _edge_list(freq)), default=1.0)

    if vmax is None:
        vals = [mean_logOR[CSV_IX[a], CSV_IX[b]]
                for a, b, _ in _edge_list(freq)
                if not np.isnan(mean_logOR[CSV_IX[a], CSV_IX[b]])]
        vmax = max((abs(v) for v in vals), default=1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu                # Blue = positive log-OR, Red = negative

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)

    for a, b, f in _edge_list(freq):
        lor = mean_logOR[CSV_IX[a], CSV_IX[b]]
        if np.isnan(lor):
            continue
        t = min(f / fmax, 1.0)
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=cmap(norm(lor)), linewidth=W_MIN + (W_MAX - W_MIN) * t,
                alpha=B_ALPHA, solid_capstyle="round", zorder=1)
    _draw_nodes(ax, pos, radii)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=-0.05, location="top", aspect=28)
    cbar.set_label("Mean conditional log-OR", fontsize=15, labelpad=10)
    if SHOW_TITLE and title:
        ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_logOR_split(freq, sign, auc, ecs, out_path, title=None, vmax=None, n_region=None):
    """One edge per SIGN: countries with positive vs negative association are drawn
    as separate red/blue lines (offset side-by-side when both occur), width
    proportional to the fraction of countries that vote that way. Uniform-sign edges
    stay single. Sign-mixed associations are annotated with their minority countries."""
    pos_cnt, neg_cnt = sign["pos_cnt"], sign["neg_cnt"]
    mean_pos, mean_neg = sign["mean_pos"], sign["mean_neg"]
    pos = _circular_pos(INDICATORS, r=1.0)
    radii = _node_radii(auc)

    counts, vals = [], []
    for a, b, _ in _edge_list(freq):
        ia, ib = CSV_IX[a], CSV_IX[b]
        if pos_cnt[ia, ib] > 0:
            counts.append(pos_cnt[ia, ib]); vals.append(mean_pos[ia, ib])
        if neg_cnt[ia, ib] > 0:
            counts.append(neg_cnt[ia, ib]); vals.append(mean_neg[ia, ib])
    # width denominator: region size (absolute, comparable) or per-figure max count
    denom = (n_region if (COMPARABLE_SCALES and n_region) else max(counts, default=1.0))
    if vmax is None:
        vmax = max((abs(v) for v in vals), default=1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu
    DELTA = 0.035                        # perpendicular offset for the two sign-edges

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)

    for a, b, _ in _edge_list(freq):
        ia, ib = CSV_IX[a], CSV_IX[b]
        P, Q = pos[a], pos[b]
        d = Q - P
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d)        # unit perpendicular
        both = pos_cnt[ia, ib] > 0 and neg_cnt[ia, ib] > 0
        for cnt, meanv, sgn in ((pos_cnt[ia, ib], mean_pos[ia, ib], +1),
                                (neg_cnt[ia, ib], mean_neg[ia, ib], -1)):
            if cnt <= 0:
                continue
            off = DELTA * sgn * nrm if both else np.zeros(2)
            t = min(cnt / denom, 1.0)
            ax.plot([P[0] + off[0], Q[0] + off[0]], [P[1] + off[1], Q[1] + off[1]],
                    color=cmap(norm(meanv)), linewidth=W_MIN + (W_MAX - W_MIN) * t,
                    alpha=B_ALPHA, solid_capstyle="round", zorder=1)
    _draw_nodes(ax, pos, radii)

    # annotate sign-mixed associations with their minority (outlier) countries
    notes = []
    for a, b, _ in sorted(_edge_list(freq), key=lambda t: -t[2]):
        lo, hi = sorted((CSV_IX[a], CSV_IX[b]))
        rec = ecs[(lo, hi)]
        npos, nneg = len(rec["pos"]), len(rec["neg"])
        if npos > 0 and nneg > 0:
            minority = rec["pos"] if npos <= nneg else rec["neg"]
            msign = "+" if npos <= nneg else "−"
            isos = ", ".join(f"{iso.upper()}({v:+.1f})" for iso, _, v in
                             sorted(minority, key=lambda r: abs(r[2]), reverse=True))
            notes.append(f"{LABELS[a]}–{LABELS[b]}: {npos}+/{nneg}−  "
                         f"outlier {msign} → {isos}")
    if SHOW_OUTLIER_NOTES and notes:
        ax.text(0.0, 0.0, "Sign-mixed associations (minority countries):\n" + "\n".join(notes),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=-0.05, location="top", aspect=28)
    cbar.set_label("Mean log-OR within sign group   (width = fraction of countries)",
                   fontsize=12, labelpad=10)
    if SHOW_TITLE and title:
        ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _build_panels():
    """[(label, allowed_isos|None, filename_suffix)] = global + one per world region."""
    panels = [("Global", None, "")]
    if PER_REGION:
        region_map = load_region_map()
        files = sorted(glob.glob(os.path.join(STABLE_DIR, f"*_{SCENARIO}_{SYM}.txt")))
        regions = {}
        for f in files:
            iso = os.path.basename(f)[:3]
            reg = region_map.get(iso)
            if reg:
                regions.setdefault(reg, set()).add(iso)
        for reg in sorted(regions, key=lambda r: -len(regions[r])):
            panels.append((reg, regions[reg], "_" + reg.replace(" ", "_")))
    return panels


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"### Global / per-region graphs ({SCENARIO}/{SYM}) ###")
    panels = _build_panels()

    # Pass 1: aggregate every panel; collect a ROBUST shared log-OR colour range
    # (95th pctile of drawn-edge |log-OR| across panels, so one extreme small-region
    #  edge doesn't wash out every figure; extremes just saturate to the end colour).
    aggs = {}
    abs_vals = []
    for label, allowed, _ in panels:
        freq, mean_logOR, auc, n, sign, ecs = aggregate(allowed)
        aggs[label] = (freq, mean_logOR, auc, n, sign, ecs)
        for a, b, _ in _edge_list(freq):
            v = mean_logOR[CSV_IX[a], CSV_IX[b]]
            if np.isfinite(v):
                abs_vals.append(abs(v))
    shared_vmax = float(np.percentile(abs_vals, 95)) if abs_vals else 1e-6
    vmax = max(shared_vmax, 1e-6) if COMPARABLE_SCALES else None
    print(f"  panels={len(panels)}  shared log-OR vmax(p95)={shared_vmax:.2f}"
          if COMPARABLE_SCALES else f"  panels={len(panels)} (per-figure scales)")

    # Pass 2: render the three variants per panel
    for label, allowed, suffix in panels:
        freq, mean_logOR, auc, n, sign, ecs = aggs[label]
        title = f"{label} (n = {n})"
        mixed = sum(1 for a, b, _ in _edge_list(freq)
                    if sign["pos_cnt"][CSV_IX[a], CSV_IX[b]] > 0
                    and sign["neg_cnt"][CSV_IX[a], CSV_IX[b]] > 0)
        print(f"\n# {label}: n={n}  edges={sum(1 for _ in _edge_list(freq))}  sign-mixed={mixed}")
        plot_gray(freq, auc, os.path.join(IMG_DIR, f"global_graph{suffix}.png"), title)
        if COMPUTE_LOGOR:
            plot_logOR(freq, mean_logOR, auc,
                       os.path.join(IMG_DIR, f"global_graph_logOR{suffix}.png"), title, vmax)
            plot_logOR_split(freq, sign, auc, ecs,
                             os.path.join(IMG_DIR, f"global_graph_logOR_split{suffix}.png"),
                             title, vmax, n_region=n)
    print("\nDone.")


if __name__ == "__main__":
    main()
