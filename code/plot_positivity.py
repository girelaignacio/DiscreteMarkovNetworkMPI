#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Country-level figures on the support of the deprivation-profile distribution,
grouped/coloured by world region (same visual style throughout).

Base figure
  images/positivity_combined_by_region.png : stacked observed-vs-zero profiles
      (left axis) + distinct/sample-size ratio (right axis).

Three "structural zeros" options (run, then pick one):
  images/discovery_opt1_pnew.png   : positivity bars + Good-Turing P(next
      household reveals a new profile) on the right axis.
  images/discovery_opt2_chao1.png  : per country, observed profiles + Chao1
      additional discoverable profiles (sampling zeros) + structural zeros,
      on the single 0..1024 count axis.
  images/discovery_opt3_rarefaction.png : profile-accumulation (rarefaction)
      curves vs number of households, saturating below 1024.

Run:  python code/plot_positivity.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROC_DIR = os.path.join(REPO, "processed_data")
LOOKUP_XLSX = os.path.join(REPO, "utils", "Table 1 National Results MPI 2024.xlsx")
IMG_DIR = os.path.join(REPO, "images")

INDICATORS = ["d_cm", "d_nutr", "d_satt", "d_educ", "d_elct",
              "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"]
NPROFILES = 2 ** len(INDICATORS)   # 1024
ZERO_COLOR = "#dcdcdc"

REGION_ORDER = [
    "Arab States", "South Asia", "Sub-Saharan Africa",
    "Latin America and the Caribbean", "East Asia and the Pacific",
    "Europe and Central Asia",
]
REGION_SHORT = {
    "Arab States": "Arab\nStates",
    "South Asia": "South\nAsia",
    "Sub-Saharan Africa": "Sub-Saharan Africa",
    "Latin America and the Caribbean": "Latin America\n& Caribbean",
    "East Asia and the Pacific": "East Asia\n& Pacific",
    "Europe and Central Asia": "Europe &\nCentral Asia",
}
REGION_COLORS = {
    "Arab States": "#c44e52",
    "South Asia": "#dd8452",
    "Sub-Saharan Africa": "#4c72b0",
    "Latin America and the Caribbean": "#55a868",
    "East Asia and the Pacific": "#8172b3",
    "Europe and Central Asia": "#937860",
}


def load_region_map():
    raw = pd.read_excel(LOOKUP_XLSX, header=None, skiprows=9)
    iso2region, iso2name = {}, {}
    for _, r in raw.iterrows():
        iso = str(r[1]).strip()
        if len(iso) == 3 and iso.isalpha():
            iso2region[iso.upper()] = str(r[3]).strip()
            iso2name[iso.upper()] = str(r[2]).strip()
    return iso2region, iso2name


def load_data():
    """Return (data DataFrame of per-country stats, codes dict iso->profile codes)."""
    iso2region, iso2name = load_region_map()
    rows, codes_d = [], {}
    w = (1 << np.arange(len(INDICATORS)))
    for f in sorted(glob.glob(os.path.join(PROC_DIR, "*"))):
        if os.path.isdir(f):
            continue
        iso = os.path.basename(f)[:3].upper()
        region = iso2region.get(iso)
        if region is None:
            print(f"  [warn] no region for {iso}; skipping"); continue
        X = pd.read_csv(f)[INDICATORS].dropna().astype(int).values
        codes = X.dot(w).astype(np.int32)             # 0..1023
        counts = np.bincount(codes, minlength=NPROFILES)
        n = int(codes.shape[0])
        S = int((counts > 0).sum())
        f1 = int((counts == 1).sum()); f2 = int((counts == 2).sum())
        chao1 = S + (f1 * f1 / (2 * f2) if f2 > 0 else f1 * (f1 - 1) / 2)
        rows.append((iso, iso2name.get(iso, iso), region, S, n, f1, f2,
                     chao1, f1 / n, S / n))
        codes_d[iso] = codes
    data = pd.DataFrame(rows, columns=["iso", "name", "region", "observed", "n",
                                       "f1", "f2", "chao1", "pnew", "ratio"])
    data["region"] = pd.Categorical(data["region"], categories=REGION_ORDER, ordered=True)
    return data, codes_d


# --------------------------------------------------------------------------- #
# layout helper: x positions grouped by region, sorted high->low within region
# --------------------------------------------------------------------------- #
def _layout(data, sortcol):
    d = data.sort_values(["region", sortcol], ascending=[True, False]).reset_index(drop=True)
    GAP = 1.4
    xs, iso, spans, idx = [], [], {}, []
    x, prev = 0.0, None
    for reg in REGION_ORDER:
        sub = d[d["region"] == reg]
        if sub.empty:
            continue
        if prev is not None:
            x += GAP
        x0 = x
        for j in sub.index:
            xs.append(x); iso.append(d.loc[j, "iso"]); idx.append(j); x += 1.0
        spans[reg] = (x0, x - 1)
        prev = reg
    return d, np.array(xs), iso, idx, spans


def _region_headers(ax, spans, ytop):
    for reg, (x0, x1) in spans.items():
        xc = (x0 + x1) / 2
        ax.plot([x0 - 0.4, x1 + 0.4], [ytop * 1.05] * 2, color="black",
                lw=3, solid_capstyle="butt", clip_on=False, zorder=5)
        ax.text(xc, ytop * 1.08, REGION_SHORT[reg], ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="black", linespacing=0.9)


def _xaxis(ax, xs, iso):
    ax.set_xlim(-1, xs[-1] + 1)
    ax.set_xticks(xs); ax.set_xticklabels(iso, rotation=90, fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=1)


# --------------------------------------------------------------------------- #
# base combined figure (bars + a right-axis series)
# --------------------------------------------------------------------------- #
def draw_combined(data, out_name, rcol, rlabel, rticks=None, rmax_frac=0.60):
    d, xs, iso, idx, spans = _layout(data, "observed")
    obs = d.loc[idx, "observed"].values.astype(float)
    rv = d.loc[idx, rcol].values.astype(float)
    colors = [REGION_COLORS[d.loc[j, "region"]] for j in idx]

    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    ax.bar(xs, obs, width=0.84, color=colors, edgecolor="white", linewidth=0.2, zorder=3)
    ax.bar(xs, NPROFILES - obs, bottom=obs, width=0.84, color=ZERO_COLOR,
           edgecolor="white", linewidth=0.2, zorder=2)
    _region_headers(ax, spans, NPROFILES)
    ax.set_ylim(0, NPROFILES * 1.22); ax.set_yticks([0, 256, 512, 768, 1024])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Number of observed deprivation profiles", fontsize=11)
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.6"); ax.spines["left"].set_bounds(0, NPROFILES)
    ax.spines["bottom"].set_color("0.6")
    _xaxis(ax, xs, iso)

    ax2 = ax.twinx()
    ax2.plot(xs, rv, color="#111111", lw=1.0, marker="o", ms=2.6, mfc="#111111",
             mec="white", mew=0.3, zorder=6)
    ax2_top = rv.max() / rmax_frac
    ax2.set_ylim(0, ax2_top)
    if rticks is not None:
        ax2.set_yticks(rticks)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_ylabel(rlabel, fontsize=11)
    ax2.spines["top"].set_visible(False); ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("0.6"); ax2.spines["right"].set_bounds(0, ax2_top / 1.22)

    handles = [Patch(facecolor=ZERO_COLOR, label="structural / sampling zeros"),
               Line2D([0], [0], color="#111111", marker="o", ms=3.5, mec="white",
                      mew=0.4, lw=1.0, label=rlabel + " (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False,
               handlelength=1.5, bbox_to_anchor=(0.5, -0.012))
    fig.subplots_adjust(left=0.085, right=0.90, top=0.99, bottom=0.16)
    fig.savefig(os.path.join(IMG_DIR, out_name), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_name}")


# --------------------------------------------------------------------------- #
# Option 2: Chao1 decomposition (observed + discoverable + structural zeros)
# --------------------------------------------------------------------------- #
def draw_chao1(data, out_name):
    d, xs, iso, idx, spans = _layout(data, "chao1")
    obs = d.loc[idx, "observed"].values.astype(float)
    chao = np.minimum(d.loc[idx, "chao1"].values.astype(float), NPROFILES)
    disc = np.clip(chao - obs, 0, None)               # extra discoverable (sampling zeros)
    zeros = NPROFILES - obs - disc                    # structural zeros

    # colour by what each segment MEANS (not by region): observed in a solid hue,
    # the Chao1 estimate in the same hue but lighter, structural zeros in a distinct hue.
    OBS_COLOR = "#2c6fb0"                              # observed profiles
    EST_COLOR = mcolors.to_rgba(OBS_COLOR, 0.38)       # Chao1 estimate (same hue, lighter)
    STRUCT_COLOR = "#dd8452"                           # structural zeros (distinct hue)

    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    ax.bar(xs, obs, width=0.84, color=OBS_COLOR, edgecolor="white", linewidth=0.2, zorder=3)
    ax.bar(xs, disc, bottom=obs, width=0.84, color=EST_COLOR, edgecolor="white",
           linewidth=0.2, zorder=3)
    ax.bar(xs, zeros, bottom=obs + disc, width=0.84, color=STRUCT_COLOR,
           edgecolor="white", linewidth=0.2, zorder=2)
    _region_headers(ax, spans, NPROFILES)
    ax.set_ylim(0, NPROFILES * 1.22); ax.set_yticks([0, 256, 512, 768, 1024])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Number of deprivation profiles", fontsize=11)
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.6"); ax.spines["left"].set_bounds(0, NPROFILES)
    ax.spines["bottom"].set_color("0.6")
    _xaxis(ax, xs, iso)

    handles = [Patch(facecolor=OBS_COLOR, label="observed profiles"),
               Patch(facecolor=EST_COLOR, label="sampling zeros"),
               Patch(facecolor=STRUCT_COLOR, label="structural zeros")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               handlelength=1.4, bbox_to_anchor=(0.5, -0.012))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.99, bottom=0.16)
    fig.savefig(os.path.join(IMG_DIR, out_name), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_name}")


# --------------------------------------------------------------------------- #
# Option 3: rarefaction / accumulation curves
# --------------------------------------------------------------------------- #
def _accumulation(codes, npts=70, K=6, seed=0):
    n = codes.shape[0]
    ms = np.unique(np.round(np.geomspace(1, n, npts)).astype(int))
    rng = np.random.default_rng(seed)
    acc = np.zeros(ms.shape[0])
    for k in range(K):
        p = codes[rng.permutation(n)]
        _, first_idx = np.unique(p, return_index=True)
        first = np.zeros(n, bool); first[first_idx] = True
        running = np.cumsum(first)
        acc += running[ms - 1]
    return ms, acc / K


def draw_rarefaction(data, codes_d, out_name):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    d = data.sort_values(["region", "n"]).reset_index(drop=True)
    for _, r in d.iterrows():
        ms, acc = _accumulation(codes_d[r["iso"]])
        ax.plot(ms, acc, color=REGION_COLORS[r["region"]], lw=0.9, alpha=0.55, zorder=3)
    ax.axhline(NPROFILES, color="0.35", lw=1.0, ls=(0, (5, 4)), zorder=4)
    ax.text(ax.get_xlim()[1], NPROFILES, r" $2^{10}=1024$ possible", va="center",
            ha="right", fontsize=9, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Number of households sampled (log scale)", fontsize=11)
    ax.set_ylabel("Distinct deprivation profiles observed", fontsize=11)
    ax.set_ylim(0, NPROFILES * 1.05)
    ax.set_yticks([0, 256, 512, 768, 1024])
    ax.tick_params(labelsize=9)
    ax.grid(color="0.92", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.6"); ax.spines["bottom"].set_color("0.6")
    handles = [Line2D([0], [0], color=REGION_COLORS[r], lw=2, label=r) for r in REGION_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=7.6, frameon=False, ncol=1)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.98, bottom=0.12)
    fig.savefig(os.path.join(IMG_DIR, out_name), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_name}")


def main():
    data, codes_d = load_data()
    print(f"  {len(data)} countries | pnew {data['pnew'].min():.4f}-{data['pnew'].max():.3f}"
          f" | chao1 {data['chao1'].min():.0f}-{data['chao1'].max():.0f}"
          f" | n {data['n'].min()}-{data['n'].max()}")

    # current paper figure (ratio on the right axis) -- kept unchanged
    draw_combined(data, "positivity_combined_by_region.png", "ratio",
                  "Distinct observed profiles / sample size", rticks=[0.0, 0.05, 0.10])

    # Option 1: P(next household reveals a new profile)  (Good-Turing f1/n)
    draw_combined(data, "discovery_opt1_pnew.png", "pnew",
                  "P(next household is a new profile)")

    # Option 2: Chao1 decomposition
    draw_chao1(data, "discovery_opt2_chao1.png")

    # Option 3: rarefaction curves
    draw_rarefaction(data, codes_d, "discovery_opt3_rarefaction.png")


if __name__ == "__main__":
    main()
