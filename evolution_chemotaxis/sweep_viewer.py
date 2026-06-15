"""
sweep_viewer.py
---------------
Loads all sweep_*_vs_*.json files from a directory and plots them
as a single figure of heatmaps arranged in a grid.

Usage
-----
    python sweep_viewer.py
    python sweep_viewer.py --in-dir sweeps/ --out summary.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot all pair sweeps in one figure")
    p.add_argument("--in-dir", default=".", help="Directory containing sweep JSON files")
    p.add_argument("--out",    default="sweep_summary.png", help="Output image path")
    p.add_argument("--cols",   type=int, default=7, help="Number of columns in the grid")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)

    files = sorted(in_dir.glob("sweep_*_vs_*.json"))
    if not files:
        print(f"No sweep JSON files found in {in_dir}")
        return

    n     = len(files)
    ncols = args.cols
    nrows = (n + ncols - 1) // ncols

    print(f"Found {n} sweeps → {nrows}×{ncols} grid")

    # shared colour scale across all panels
    all_vals = []
    sweeps = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        grid = np.array(d["grid"])
        all_vals.extend(grid[~np.isnan(grid)].tolist())
        sweeps.append(d)

    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.8 * ncols, 2.5 * nrows),
                             squeeze=False)

    for ax_idx, (d, ax) in enumerate(zip(sweeps, axes.flat)):
        grid = np.array(d["grid"])
        axis = np.array(d["axis"])
        p1, p2 = d["p1"], d["p2"]
        base   = d.get("base_params", {})

        im = ax.imshow(
            grid,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
        )

        # mark base values
        if p1 in base:
            ax.axhline(base[p1], color="red", lw=0.8, ls="--", alpha=0.7)
        if p2 in base:
            ax.axvline(base[p2], color="red", lw=0.8, ls="--", alpha=0.7)

        # short labels (strip common suffix for readability)
        def short(name):
            return name.replace("_minus", "⁻").replace("p_", "p").replace("a_", "a")

        ax.set_xlabel(short(p2), fontsize=6)
        ax.set_ylabel(short(p1), fontsize=6)
        ax.tick_params(labelsize=5)

        # best point marker
        best_ij = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.scatter(axis[best_ij[1]], axis[best_ij[0]],
                   marker="*", color="white", s=40, zorder=5)

    # hide unused axes
    for ax in axes.flat[len(sweeps):]:
        ax.set_visible(False)

    # shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.5, wspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Wasserstein fitness (lower = better)")

    plt.suptitle("Fitness landscape — all parameter pairs", fontsize=11, y=1.01)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()