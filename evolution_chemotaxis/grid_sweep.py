"""
grid_sweep.py
-------------
Coarse 2D grid sweep over any two parameters, keeping the rest fixed.
Produces a heatmap of the fitness landscape.

Usage
-----
    python grid_sweep.py --p1 p1_minus --p2 a1
    python grid_sweep.py --p1 p1_minus --p2 a_run_rev --n 7 --base best_params.json
"""

import argparse
import json
import logging
import os
import subprocess
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

# ── constants (keep in sync with optimize_params.py) ─────────────────────────

SIM_DIR         = Path("~/cuda_agent_based_sim").expanduser()
SIM_SCRIPT      = SIM_DIR / "offline_build_and_run.sh"
PARAMS_JSON     = SIM_DIR / "params.json"
SIM_OUTPUT_JSON = SIM_DIR / "auto_agents_100_all_data.json"
REAL_STATS_JSON = "real_worm_stats.json"

HIT_DIST_MM = 5.0
T_START     = 10.0
DT          = 1 / 3
W_BEFORE    = 1.0
W_AFTER     = 0.0
SIM_TIMEOUT = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── baseline parameters ───────────────────────────────────────────────────────

BASE_PARAMS = {
    "p1_minus":          0.24699125242443848,
    "a1":                0.3045226993793369,
    "p_run_rev_minus":   0.5878576636846718,
    "p_run_turn_minus":  0.5,
    "p_rev_run_minus":   0.5390861921044875,
    "p_rev_turn_minus":  0.5,
    "p_turn_run_minus":  0.3139107776571001,
    "p_turn_rev_minus":  0.5,
    "a_run_rev":         0.2410485591972853,
    "a_run_turn":        0.1,
    "a_rev_run":         0.5422182324571516,
    "a_rev_turn":        0.1,
    "a_turn_run":        0.45589235436573516,
    "a_turn_rev":        0.1,
}

# ── helpers (same as optimize_params.py) ─────────────────────────────────────

def diffusion_profile_distance(r, t, D=1.0, Q=1.0):
    if t <= 0:
        return 0.0
    return Q / (4 * np.pi * D * t) * np.exp(-(r ** 2) / (4 * D * t))


def compute_C_trace(xy_mm, odor_orig, t_start=T_START, dt=DT):
    distance = np.linalg.norm(xy_mm - np.array(odor_orig), axis=1)
    C = np.array([
        diffusion_profile_distance(distance[i], t_start + i * dt)
        for i in range(len(xy_mm))
    ])
    dC = np.diff(C, prepend=C[0])
    return C, dC, distance


def extract_hit_stats(data):
    positions = np.array(data["positions"])
    params    = data["parameters"]
    odor_orig = (params["ODOR_X0"], params["ODOR_Y0"])
    n_agents, t_max, _ = positions.shape
    before_fracs, after_fracs = [], []
    for i in range(n_agents):
        _, dC, d = compute_C_trace(positions[i], odor_orig)
        hit_idxs = np.where(d < HIT_DIST_MM)[0]
        if hit_idxs.size == 0:
            continue
        hit_idx = int(hit_idxs[0])
        if hit_idx == 0:
            continue
        before_fracs.append(float(np.sum(dC[:hit_idx] > 0) / hit_idx))
        frames_after = t_max - hit_idx
        if frames_after > 0:
            after_fracs.append(float(np.sum(dC[hit_idx:] > 0) / frames_after))
    return before_fracs, after_fracs


def run_sim_and_score(params_dict, real_before, real_after):
    with open(PARAMS_JSON, "w") as f:
        json.dump(params_dict, f, indent=2)

    env = os.environ.copy()
    try:
        result = subprocess.run(
            [str(SIM_SCRIPT)],
            env=env,
            cwd=SIM_DIR,
            timeout=SIM_TIMEOUT,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("Simulation timed out")
        return np.nan
    except Exception as e:
        log.warning("Simulation failed: %s", e)
        return np.nan

    if result.returncode != 0:
        log.warning("Non-zero exit:\n%s", result.stderr[-300:])
        return np.nan

    try:
        with open(SIM_OUTPUT_JSON) as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Could not read output: %s", e)
        return np.nan

    sim_before, sim_after = extract_hit_stats(data)
    if len(sim_before) < 2:
        log.warning("Too few hitting agents (%d) — penalising", len(sim_before))
        return 1.0

    score = W_BEFORE * wasserstein_distance(real_before, sim_before)
    if W_AFTER > 0 and len(sim_after) >= 2:
        score += W_AFTER * wasserstein_distance(real_after, sim_after)
    return score


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="2D grid sweep of fitness landscape")
    p.add_argument("--p1",   required=True, help="First parameter to vary")
    p.add_argument("--p2",   required=True, help="Second parameter to vary")
    p.add_argument("--n",    type=int, default=5, help="Grid points per axis (default 5)")
    p.add_argument("--base", default=None,
                   help="JSON file with base params (overrides hardcoded defaults)")
    p.add_argument("--out",  default=None,
                   help="Save heatmap to this path (e.g. sweep.png); shown interactively if omitted")
    return p.parse_args()


def main():
    args = parse_args()

    # load base params
    base = BASE_PARAMS.copy()
    if args.base:
        with open(args.base) as f:
            base.update({k: v for k, v in json.load(f).items() if k in base})
        log.info("Base params loaded from %s", args.base)

    # validate
    for name in (args.p1, args.p2):
        if name not in base:
            raise ValueError(f"Unknown parameter '{name}'. Valid: {list(base.keys())}")
    if args.p1 == args.p2:
        raise ValueError("--p1 and --p2 must be different parameters")

    # load real stats
    with open(REAL_STATS_JSON) as f:
        d = json.load(f)
    real_before = np.array(d["before"])
    real_after  = np.array(d["after"])
    log.info("Real stats: before n=%d  after n=%d", len(real_before), len(real_after))

    # build grid
    axis = np.linspace(0.0, 1.0, args.n)
    grid = np.full((args.n, args.n), np.nan)

    total = args.n * args.n
    for idx, (i, j) in enumerate(product(range(args.n), range(args.n))):
        v1, v2 = axis[i], axis[j]
        params = base.copy()
        params[args.p1] = v1
        params[args.p2] = v2

        log.info("[%d/%d]  %s=%.3f  %s=%.3f",
                 idx + 1, total, args.p1, v1, args.p2, v2)

        score = run_sim_and_score(params, real_before, real_after)
        grid[i, j] = score
        log.info("  → fitness = %.4f", score)

    # save raw results
    results = {
        "p1": args.p1, "p2": args.p2,
        "axis": axis.tolist(),
        "grid": grid.tolist(),
        "base_params": base,
    }
    results_path = f"sweep_{args.p1}_vs_{args.p2}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Raw results saved to %s", results_path)

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis_r",   # reversed: darker = better (lower fitness)
    )
    plt.colorbar(im, ax=ax, label="Wasserstein fitness (lower = better)")
    ax.set_xlabel(args.p2)
    ax.set_ylabel(args.p1)
    ax.set_title(f"Fitness landscape\n{args.p1} vs {args.p2}")

    # annotate cells
    for i, j in product(range(args.n), range(args.n)):
        val = grid[i, j]
        if not np.isnan(val):
            ax.text(
                axis[j], axis[i], f"{val:.3f}",
                ha="center", va="center", fontsize=7,
                color="white" if val < np.nanmedian(grid) else "black"
            )

    # mark the base value of each parameter
    ax.axhline(base[args.p1], color="red", lw=1, ls="--", alpha=0.6, label=f"base {args.p1}")
    ax.axvline(base[args.p2], color="red", lw=1, ls="--", alpha=0.6, label=f"base {args.p2}")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        log.info("Heatmap saved to %s", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()