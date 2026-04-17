"""
Compute log-odds ratios for every country whose stable network is stored in
results_stable2/ (mpi_poor_conserv graphs — the same ones used in contribution.py).

For each country we compute:
  - logOR_matrix[i,j]: conditional logOR of X_i and X_j given Y and the rest of
                        the neighbourhood X_W\{j}  (only for selected edges).
  - logOR_Y[i]:         conditional logOR of X_i and Y given X_W_i.

Both quantities are computed by calling discrete_graphical_model.compute_interaction_logOR().

Results are saved to ../results_logOR/<filename>_logOR_summary.txt (human-readable)
and the raw matrices as <filename>_logOR_matrix.txt / <filename>_logOR_Y.txt.

Run this script from the code/ directory:
    cd code && python compute_logOR.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_gm_nonpos import discrete_graphical_model

# ---------------------------------------------------------------------------
# MPI helpers (same as estimate_networks.py / contribution.py)
# ---------------------------------------------------------------------------
dimensions_indicators = {
    "hl": ["d_cm", "d_nutr"],
    "ed": ["d_satt", "d_educ"],
    "ls": ["d_elct", "d_wtr", "d_sani", "d_hsg", "d_ckfl", "d_asst"],
}


def calculate_weights(mpi_indicators):
    dim_weights, indic_weights = {}, {}
    for key, vals in mpi_indicators.items():
        w = 1 / len(mpi_indicators)
        dim_weights[key] = w
        for v in vals:
            indic_weights[v] = w / len(vals)
    return dim_weights, indic_weights


def deprivation_score(mpi_indicators, data):
    _, iw = calculate_weights(mpi_indicators)
    mpi_data = data[list(iw.keys())].copy()
    for col in mpi_data.columns:
        mpi_data[col] *= iw[col]
    return mpi_data.sum(axis=1)


def censored_deprivation_score(score, k):
    return np.where(score >= k, score, 0)


# ---------------------------------------------------------------------------
# Paths (relative to code/ working directory)
# ---------------------------------------------------------------------------
PROC_DIR   = "../processed_data"
STABLE_DIR = "../results_stable2"
OUT_DIR    = "../results_logOR"

os.makedirs(OUT_DIR, exist_ok=True)

# Dummy instance — compute_interaction_logOR uses no instance attributes
dgm = discrete_graphical_model()

filenames = sorted(os.listdir(PROC_DIR))
total = len(filenames)

for counter, filename in enumerate(filenames, 1):
    summary_path = os.path.join(OUT_DIR, f"{filename}_logOR_summary.txt")
    if os.path.exists(summary_path):
        print(f"[{counter}/{total}] {filename} — already done, skipping.")
        continue

    adj_path = os.path.join(STABLE_DIR, f"{filename}_mpi_poor_conserv.txt")
    if not os.path.exists(adj_path):
        print(f"[{counter}/{total}] {filename} — adj matrix not found, skipping.")
        continue

    print(f"[{counter}/{total}] Computing logOR for {filename} …")

    # Load and clean data
    df = pd.read_csv(os.path.join(PROC_DIR, filename), index_col=0)
    df = df.dropna().astype(int)
    col_names = df.columns.tolist()  # e.g. [d_cm, d_nutr, d_satt, ...]

    # Build Y (mpi_poor indicator)
    c_k = censored_deprivation_score(
        deprivation_score(dimensions_indicators, df), 33 / 100
    )
    mpi_poor = np.where(c_k > 0, 1, 0).reshape(-1, 1)

    X = df.to_numpy().astype(int)
    Y = mpi_poor.astype(int)

    # Load adjacency matrix (same graph used in contribution.py)
    adj = np.loadtxt(adj_path).astype(int)

    # Compute logOR (conditional on Y)
    logOR_matrix, logOR_Y = dgm.compute_interaction_logOR(
        X, Y, ne=adj, smoothing=1.0, symmetrize=True
    )

    # Save raw matrices
    np.savetxt(
        os.path.join(OUT_DIR, f"{filename}_logOR_matrix.txt"),
        logOR_matrix, fmt="%.6f"
    )
    np.savetxt(
        os.path.join(OUT_DIR, f"{filename}_logOR_Y.txt"),
        logOR_Y.reshape(1, -1), fmt="%.6f"
    )

    # Save human-readable summary
    lines = []
    lines.append(f"=== logOR for {filename} ===\n")
    lines.append(f"Graph: results_stable2/{filename}_mpi_poor_conserv.txt\n")
    lines.append(f"Outcome: MPI-poor (censored deprivation score >= 1/3)\n")
    lines.append(f"Column order: {col_names}\n\n")

    lines.append("--- Pairwise conditional logOR: X_i -- X_j  (given Y and neighbourhood X_W\\j) ---\n")
    p = len(col_names)
    has_edges = False
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] == 1 and not np.isnan(logOR_matrix[i, j]):
                lines.append(
                    f"  {col_names[i]:8s}  --  {col_names[j]:8s} :  logOR = {logOR_matrix[i, j]:+.4f}\n"
                )
                has_edges = True
    if not has_edges:
        lines.append("  (no selected edges)\n")

    lines.append(
        "\n--- logOR of each indicator with Y (MPI-poor), given neighbourhood X_W ---\n"
    )
    for i in range(p):
        lines.append(
            f"  {col_names[i]:8s} :  logOR_Y = {logOR_Y[i]:+.4f}\n"
        )

    with open(summary_path, "w") as fh:
        fh.writelines(lines)

print("\nDone. Results saved to", OUT_DIR)
