#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path-only versions of the per-country stability figures (no Pi-matrix heatmap),
for the supplement (Fig. 5 / "Stability of the estimated edges"). Reuses the
left-panel drawing routine from plot_stability_paths.py.

Output: images/<country>_<scenario>_<sym>_stability_pathonly.png

Run:  python code/plot_stability_paths_only.py
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_stability_paths import (  # noqa: E402  (guarded by __main__, safe to import)
    _load, _draw_path, COUNTRIES, SCENARIOS, SYMMETRIZATIONS, SYM_IX, IMG_DIR, DPI,
)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        for sym in SYMMETRIZATIONS:
            sym_ix = SYM_IX[sym]
            for country, disp in COUNTRIES.items():
                d = _load(country, scenario)
                if d is None or int(d["index_selected"][sym_ix]) < 0:
                    print(f"  [skip] {country}/{scenario}/{sym}")
                    continue
                fig, ax = plt.subplots(figsize=(6.4, 5.0))
                _draw_path(ax, d, sym_ix, list(d["indicator_names"]))
                ax.set_title(disp, fontsize=13, fontweight="bold")
                fig.tight_layout()
                out = os.path.join(
                    IMG_DIR, f"{country}_{scenario}_{sym}_stability_pathonly.png")
                fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"  wrote {out}")


if __name__ == "__main__":
    main()
