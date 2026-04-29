"""
Plots the optimized L1 and L2 logistic functions from l1_aggregation.json,
l1_diffusion.json, l2_aggregation.json, l2_diffusion.json.

Run from the evolution/ subfolder:
    python plot_functions.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
STATE_EST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "state_estimations"))

STATES      = [0, 1, 2]
TRANSITIONS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

STATE_LABELS = {0: "State 0", 1: "State 1", 2: "State 2"}

# ─── Aesthetics ───────────────────────────────────────────────────────────────

DARK_BG   = "#0e0e12"
PANEL_BG  = "#16161e"
GRID_COL  = "#2a2a3a"
TEXT_COL  = "#e0e0f0"
SUB_COL   = "#7070a0"

LOG_DIR   = "logs"


# Two behaviour palettes
PALETTE = {
    "aggregation": "#f4a261",   # warm amber
    "diffusion":   "#48cae4",   # cool cyan
}

STATE_COLORS = ["#c77dff", "#80ffdb", "#ff6b6b"]  # per state

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       TEXT_COL,
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      SUB_COL,
    "ytick.color":      SUB_COL,
    "axes.facecolor":   PANEL_BG,
    "figure.facecolor": DARK_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.grid":        True,
    "grid.color":       GRID_COL,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.8,
})

# ─── Helpers ──────────────────────────────────────────────────────────────────

def logistic(N, coeff, intercept, height):
    return height / (1.0 + np.exp(-(coeff * N + intercept)))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_behaviour(label):
    """Returns (l1_data, l2_data) for a given behaviour label."""
    l1 = load_json(os.path.join(STATE_EST_DIR, f"l1_{label}.json"))
    l2 = load_json(os.path.join(STATE_EST_DIR, f"l2_{label}.json"))
    return l1, l2


def get_l1_params(l1_data, state):
    entry = l1_data[str(state)]
    return entry["model_coeff"], entry["model_intercept"], entry["model_height"]


def get_l2_params(l2_data, src, dst):
    entry = l2_data[str(src)][str(dst)]
    return entry["model_coeff"], entry["model_intercept"], entry["model_height"]


def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=8, pad=6, fontweight="bold")
    ax.set_xlabel("N neighbors", fontsize=7, color=SUB_COL)
    ax.set_ylabel("probability", fontsize=7, color=SUB_COL)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─── Main plot ────────────────────────────────────────────────────────────────
import glob
def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "AGGREGATION_iter*_cluster_fractions.json")))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    cmap = plt.cm.plasma
    iterations = []
    for i, path in enumerate(logs):
        with open(path) as f:
            d = json.load(f)
        iterations.append(d["iteration"])
        color = cmap(i / max(len(logs) - 1, 1))
        ax.plot(d["fractions"], color=color, linewidth=1.2, alpha=0.85,
                label=f"iter {d['iteration']:04d}  fit={d['fitness']:.4f}")

    ax.set_xlabel("timestep", color=TEXT_COL, fontsize=9)
    ax.set_ylabel("largest cluster / N", color=TEXT_COL, fontsize=9)
    ax.set_title("Largest cluster fraction over time — AGGREGATION", color=TEXT_COL, fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=iterations[0], vmax=iterations[-1]) if iterations else plt.Normalize()
    )
    fig.colorbar(sm, ax=ax, label="iteration progress").ax.yaxis.label.set_color(TEXT_COL)

    ax.legend(fontsize=6, loc="upper left", framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL,
              ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "cluster_fractions_progress.png"), dpi=150, facecolor=DARK_BG)
    plt.show()


    behaviours = ["aggregation", "diffusion"]
    N = np.linspace(0, 20, 300)

    data = {}
    for b in behaviours:
        try:
            data[b] = load_behaviour(b)
        except FileNotFoundError as e:
            print(f"[WARN] Missing file for {b}: {e}")

    for b, (l1_data, l2_data) in data.items():
        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor(DARK_BG)
        fig.suptitle(
            f"PFSM Logistic Functions — {b.upper()}",
            fontsize=13, color=TEXT_COL, fontweight="bold", y=0.98,
        )

        outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                                  top=0.93, bottom=0.07, left=0.06, right=0.97)
        l1_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)
        l2_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.35)

        # L1
        for s_idx, state in enumerate(STATES):
            ax = fig.add_subplot(l1_grid[s_idx])
            style_ax(ax, f"L1 · Exit probability · {STATE_LABELS[state]}")
            coeff, intercept, height = get_l1_params(l1_data, state)
            ax.plot(N, logistic(N, coeff, intercept, height),
                    color=PALETTE[b], linewidth=2.0)
            ax.axhspan(0, 1, alpha=0.03, color=STATE_COLORS[s_idx])

        # L2
        for s_idx, src in enumerate(STATES):
            ax = fig.add_subplot(l2_grid[s_idx])
            style_ax(ax, f"L2 · Transition probability · from {STATE_LABELS[src]}")
            for dst in STATES:
                coeff, intercept, height = get_l2_params(l2_data, src, dst)
                ax.plot(N, logistic(N, coeff, intercept, height),
                        color=STATE_COLORS[dst],
                        linewidth=2.2 if src == dst else 1.4,
                        linestyle="--" if src == dst else "-",
                        label=f"→ S{dst}")
            ax.axhspan(0, 1, alpha=0.03, color=STATE_COLORS[s_idx])
            ax.legend(fontsize=6, loc="upper left", framealpha=0.3,
                      facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

        fig.text(0.01, 0.72, "L1", fontsize=16, color=SUB_COL,
                 fontweight="bold", va="center", rotation=90)
        fig.text(0.01, 0.28, "L2", fontsize=16, color=SUB_COL,
                 fontweight="bold", va="center", rotation=90)

        out_path = os.path.join(SCRIPT_DIR, f"logistic_functions_{b}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved → {out_path}")
        plt.show()


if __name__ == "__main__":
    main()