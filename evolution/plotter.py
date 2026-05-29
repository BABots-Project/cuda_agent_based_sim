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

DARK_BG   = "#ffffff"
PANEL_BG  = "#ffffff"
GRID_COL  = "#cccccc"
TEXT_COL  = "#111111"
SUB_COL   = "#444444"

LOG_DIR   = "server_results/logs"


# Two behaviour palettes
PALETTE = {
    "aggregation": "#f4a261",   # warm amber
    "diffusion":   "#48cae4",   # cool cyan
    "server":     "#ff6b6b",   # vibrant red
}

STATE_DISPLAY_LABELS = {
    0: "Reversal",
    1: "Turn",
    2: "Run",
}

STATE_COLORS = ["#c77dff", "#2a7db5", "#ff6b6b"]  # S0: purple, S1: steel blue, S2: red

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        18,
    "text.color":       TEXT_COL,
    "axes.labelcolor":  TEXT_COL,
    "axes.labelsize":   18,
    "axes.titlesize":   20,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "axes.facecolor":   PANEL_BG,
    "figure.facecolor": PANEL_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.grid":        True,
    "grid.color":       GRID_COL,
    "grid.linewidth":   0.8,
    "grid.alpha":       0.8,
    "legend.fontsize":  14,
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


def style_ax(ax, title, ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=20, pad=6, fontweight="bold")
    ax.set_xlabel(r"$\Delta n_p$", fontsize=18, color=TEXT_COL)
    ax.set_ylabel(ylabel, fontsize=18, color=TEXT_COL)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─── Main plot ────────────────────────────────────────────────────────────────
import glob
def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "DIFFUSION_iter*_cluster_fractions.json")))

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
    ax.set_title("Largest cluster fraction over time — DIFFUSION", color=TEXT_COL, fontsize=11)
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


    behaviours = ["aggregation"]#"server"]#["aggregation", "diffusion"]
    N = np.linspace(-20, 20, 300)

    data = {}
    for b in behaviours:
        try:
            data[b] = load_behaviour(b)
        except FileNotFoundError as e:
            print(f"[WARN] Missing file for {b}: {e}")

    for b, (l1_data, l2_data) in data.items():

        # ── L1 subplots ───────────────────────────────────────────────────────
        for s_idx, state in enumerate(STATES):
            if state != 2:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor(PANEL_BG)
            style_ax(ax, f"L1 · {STATE_DISPLAY_LABELS[state]}")
            coeff, intercept, height = get_l1_params(l1_data, state)
            ax.plot(N, logistic(N, coeff, intercept, height),
                    color=PALETTE[b], linewidth=2.5)

            plt.tight_layout()
            out = os.path.join(SCRIPT_DIR, f"{b}_l1_state{state}.pdf")
            plt.savefig(out, bbox_inches="tight", facecolor=PANEL_BG, dpi=300)
            print(f"Saved → {out}")
            plt.close(fig)

        # ── L2 subplots ───────────────────────────────────────────────────────
        for s_idx, src in enumerate(STATES):
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor(PANEL_BG)
            style_ax(ax, f"L2 · from {STATE_DISPLAY_LABELS[src]}", ylabel="")
            for dst in STATES:
                coeff, intercept, height = get_l2_params(l2_data, src, dst)
                ax.plot(N, logistic(N, coeff, intercept, height),
                        color=STATE_COLORS[dst],
                        linewidth=2.5,
                        linestyle="--" if src == dst else "-",
                        #label=f"→ {STATE_DISPLAY_LABELS[dst]}  m={coeff:.2f} q={intercept:.2f} h={height:.2f}")
                        label=f"→ {STATE_DISPLAY_LABELS[dst]}")
            ax.legend(fontsize=9,  framealpha=0.3,
                      facecolor=PANEL_BG, edgecolor=GRID_COL,
                      labelcolor=TEXT_COL,
                      prop={'family': 'monospace', 'size': 9})
            plt.tight_layout()
            out = os.path.join(SCRIPT_DIR, f"{b}_l2_from{src}.pdf")
            plt.savefig(out, bbox_inches="tight", facecolor=PANEL_BG, dpi=300)
            print(f"Saved → {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()