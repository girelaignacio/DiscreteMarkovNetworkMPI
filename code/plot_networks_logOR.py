"""
Generate two sets of network plots for selected countries, encoding logOR on edges.

Set 1 – network graph only:
  Edge colour and width reflect the pairwise conditional logOR(X_i, X_j | Y, X_W\j).
  Red = positive association (co-deprivation); Blue = negative.

Set 2 – network + MPI outcome node:
  Same as Set 1, plus a central "MPI" node connected to every indicator.
  Those spoke edges are coloured/sized by logOR_Y[i] = logOR(X_i, Y | X_W_i).

Images are saved to ../images/.

Run from the code/ directory:
    cd code && python plot_networks_logOR.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Indicator metadata (order + labels + colours matching app.R)
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

NODE_COLORS = {
    "d_nutr": "#962b21", "d_cm": "#652525",
    "d_educ": "#c6a9ab", "d_satt": "#a68580",
    "d_ckfl": "#afc4d1", "d_sani": "#7d9eb6",
    "d_wtr":  "#5e8199", "d_elct": "#3f6781",
    "d_hsg":  "#174d68", "d_asst": "#00384f",
}

OUTCOME_COLOR  = "#FFFFFF"   # white for outcome node
OUTCOME_LABEL  = "Z"

# ---------------------------------------------------------------------------
# Countries to plot  (iso_survey → display title)
# ---------------------------------------------------------------------------
COUNTRIES = {
    "cog_mics14-15": "Congo (DR)",
    "gnb_mics18-19": "Guinea-Bissau",
    "hnd_mics19":    "Honduras",
    "mli_dhs18":     "Mali",
    "zwe_mics19":    "Zimbabwe",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC_DIR   = "../processed_data"
STABLE_DIR = "../results_stable2"
LOGOR_DIR  = "../results_logOR"
IMG_DIR    = "../images"

os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MPI helpers
# ---------------------------------------------------------------------------
_dimensions_indicators = {
    "hl": ["d_cm", "d_nutr"],
    "ed": ["d_satt", "d_educ"],
    "ls": ["d_elct", "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"],
}


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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _circular_pos(nodes, r=1.0, start_angle=None):
    """Evenly spaced circular positions starting at top (pi/2)."""
    n = len(nodes)
    if start_angle is None:
        start_angle = np.pi / 2
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
    return {nd: np.array([r * np.cos(a), r * np.sin(a)]) for nd, a in zip(nodes, angles)}


# ---------------------------------------------------------------------------
# Core plot routine
# ---------------------------------------------------------------------------
def _plot(
    adj, logOR_matrix, logOR_Y, col_names,
    title, out_path, add_outcome=False,
    node_radius=0.13,
):
    """
    Draw the Markov-network graph with logOR-encoded edges.

    Parameters
    ----------
    adj          : (p,p) int array — adjacency from results_stable2
    logOR_matrix : (p,p) float — pairwise conditional logOR (NaN for non-edges)
    logOR_Y      : (p,)  float — per-indicator logOR with MPI outcome
    col_names    : list[str] — column order matching adj rows/cols
    add_outcome  : if True, draw central MPI node with spoke edges coloured by logOR_Y
    """
    p = len(INDICATORS)

    # Map col_names → index in adj/logOR arrays
    ci = {c: i for i, c in enumerate(col_names)}

    # Collect all displayed logOR values for shared normalisation
    edge_vals = []
    for i in range(p):
        for j in range(i + 1, p):
            ni, nj = INDICATORS[i], INDICATORS[j]
            if ni in ci and nj in ci:
                ii, jj = ci[ni], ci[nj]
                if adj[ii, jj] == 1 and not np.isnan(logOR_matrix[ii, jj]):
                    edge_vals.append(logOR_matrix[ii, jj])
    if add_outcome:
        for i in range(p):
            ni = INDICATORS[i]
            if ni in ci and not np.isnan(logOR_Y[ci[ni]]):
                edge_vals.append(logOR_Y[ci[ni]])

    if not edge_vals:
        print(f"  No edges — skipping {out_path}")
        return

    vmax = max(abs(min(edge_vals)), abs(max(edge_vals)), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu   # Blue = positive logOR, Red = negative

    lw_min, lw_max = 5., 10.0

    def _lw(v):
        return lw_min + (lw_max - lw_min) * abs(v) / vmax

    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)

    pos_ind = _circular_pos(INDICATORS, r=1.0)
    pos_all = dict(pos_ind)
    if add_outcome:
        pos_all[OUTCOME_LABEL] = np.array([0.0, 0.0])

    # --- edges between indicators ---
    for i in range(p):
        for j in range(i + 1, p):
            ni, nj = INDICATORS[i], INDICATORS[j]
            if ni in ci and nj in ci:
                ii, jj = ci[ni], ci[nj]
                if adj[ii, jj] == 1 and not np.isnan(logOR_matrix[ii, jj]):
                    lor = logOR_matrix[ii, jj]
                    ax.plot(
                        [pos_all[ni][0], pos_all[nj][0]],
                        [pos_all[ni][1], pos_all[nj][1]],
                        color=cmap(norm(lor)),
                        linewidth=_lw(lor),
                        solid_capstyle="round",
                        zorder=1, alpha=0.9,
                    )

    # --- spoke edges to MPI outcome node ---
    if add_outcome:
        for i in range(p):
            ni = INDICATORS[i]
            if ni in ci and not np.isnan(logOR_Y[ci[ni]]):
                lor = logOR_Y[ci[ni]]
                ax.plot(
                    [pos_all[ni][0], 0.0],
                    [pos_all[ni][1], 0.0],
                    color=cmap(norm(lor)),
                    linewidth=_lw(lor),
                    linestyle="--",
                    zorder=1, alpha=0.85,
                )

    # --- indicator nodes (drawn on top of edges) ---
    for ind in INDICATORS:
        x, y = pos_all[ind]
        circle = plt.Circle(
            (x, y), node_radius,
            facecolor=NODE_COLORS[ind], zorder=2,
            linewidth=1.5, edgecolor="white",
        )
        ax.add_patch(circle)
        ax.text(
            x, y, LABELS[ind],
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white", zorder=3,
        )

    # --- MPI outcome node ---
    if add_outcome:
        circle = plt.Circle(
            (0.0, 0.0), node_radius * 1.15,
            facecolor=OUTCOME_COLOR, zorder=2,
            linewidth=2.0, edgecolor="#444444",
        )
        ax.add_patch(circle)
        ax.text(
            0.0, 0.0, OUTCOME_LABEL,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="#333333", zorder=3,
        )

    # --- legend: dimensions ---
    dim_patches = [
        mpatches.Patch(color="#7d1919", label="Health"),
        mpatches.Patch(color="#b69690", label="Education"),
        mpatches.Patch(color="#5e8199", label="Living Standards"),
    ]
    if add_outcome:
        dim_patches.append(
            mpatches.Patch(facecolor=OUTCOME_COLOR, label="MPI outcome", edgecolor="#444444")
        )
    # ax.legend(
    #     handles=dim_patches,
    #     loc="upper right",
    #     fontsize=9,
    #     framealpha=0.85,
    #     edgecolor="#cccccc",
    # )

    # --- colourbar ---
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=-0.05, location="top", aspect=28)
    cbar.set_label("Conditional log-OR", fontsize=15, labelpad=10)

    edge_note = (
        "Edge colour/width: pairwise logOR | X_i\u2013X_j\n"
        if not add_outcome
        else "Solid: pairwise logOR | X_i\u2013X_j   Dashed: logOR | X_i\u2013MPI"
    )
    # ax.set_title(f"{title}\n{edge_note}", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for filename, display_name in COUNTRIES.items():
    print(f"\n--- {display_name} ({filename}) ---")

    # Paths
    data_path  = os.path.join(PROC_DIR, filename)
    adj_path   = os.path.join(STABLE_DIR, f"{filename}_mpi_poor_conserv.txt")
    lor_mat    = os.path.join(LOGOR_DIR, f"{filename}_logOR_matrix.txt")
    lor_y_path = os.path.join(LOGOR_DIR, f"{filename}_logOR_Y.txt")

    # Check pre-computed logOR; recompute if missing
    if not os.path.exists(lor_mat) or not os.path.exists(lor_y_path):
        print("  logOR files not found — computing now…")
        from discrete_gm_nonpos import discrete_graphical_model

        df = pd.read_csv(data_path, index_col=0).dropna().astype(int)
        mpi_poor = _mpi_poor(df).reshape(-1, 1)
        X = df.to_numpy().astype(int)
        Y = mpi_poor
        adj = np.loadtxt(adj_path).astype(int)

        dgm = discrete_graphical_model()
        logOR_matrix, logOR_Y = dgm.compute_interaction_logOR(
            X, Y, ne=adj, smoothing=1.0, symmetrize=True
        )
        os.makedirs(LOGOR_DIR, exist_ok=True)
        np.savetxt(lor_mat, logOR_matrix, fmt="%.6f")
        np.savetxt(lor_y_path, logOR_Y.reshape(1, -1), fmt="%.6f")
    else:
        df = pd.read_csv(data_path, index_col=0).dropna().astype(int)
        adj         = np.loadtxt(adj_path).astype(int)
        logOR_matrix = np.loadtxt(lor_mat)
        logOR_Y      = np.loadtxt(lor_y_path).ravel()

    col_names = df.columns.tolist()

    # --- Set 1: network only ---
    out1 = os.path.join(IMG_DIR, f"{filename}_logOR_network.png")
    _plot(
        adj, logOR_matrix, logOR_Y, col_names,
        title=display_name,
        out_path=out1,
        add_outcome=False,
    )

    # --- Set 2: network + MPI outcome node ---
    out2 = os.path.join(IMG_DIR, f"{filename}_logOR_network_with_outcome.png")
    _plot(
        adj, logOR_matrix, logOR_Y, col_names,
        title=f"{display_name} — with MPI outcome",
        out_path=out2,
        add_outcome=True,
    )

print("\nAll plots saved to", IMG_DIR)
