"""
CMA-ES optimization of PFSM logistic parameters (L1 and L2).
Fully parallelised: all candidates x ensemble runs execute simultaneously.

Parameter vector layout (30 values total, normalized to [-1, 1]):
  [0:3]   L1  — state 2 only (coeff, intercept, height)
  [3:30]  L2  — 9 transitions x 3 params (coeff, intercept, height)

Transitions order: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
"""

import json
import subprocess
import os
import shutil
import numpy as np
import cma
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import signal
import sys

def _cleanup(sig, frame):
    print("\n[INTERRUPTED] Killing all docker containers...")
    os.system("docker ps -q | xargs docker kill 2>/dev/null")
    sys.exit(1)

signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATE_EST_DIR  = os.path.join(SIM_ROOT, "state_estimations")
SIM_OUTPUT_DIR = os.path.join(SIM_ROOT, "outputs")   # base dir for all runs
RUN_SCRIPT     = os.path.join(SIM_ROOT, "offline_build_and_run_parallel.sh")

L1_PATH       = os.path.join(STATE_EST_DIR, "l1.json")
L2_PATH       = os.path.join(STATE_EST_DIR, "l2.json")
OFF_FOOD_PATH = os.path.join(STATE_EST_DIR, "off_food_transitions.json")

STATES      = [0, 1, 2]
L1_STATES   = [2]
TRANSITIONS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

NEIGHBOR_RADIUS_MM = 0.5
EPS                = 1e-6
DOMAIN_W           = 10.0
DOMAIN_H           = 10.0
N_ENSEMBLE         = 5
LOG_EVERY          = 20
LOG_DIR            = os.path.join(SCRIPT_DIR, "logs")

# ─── Parameter scaling ────────────────────────────────────────────────────────

COEFF_RANGE     = (-5.0,  5.0)
INTERCEPT_RANGE = (-20.0, 20.0)
HEIGHT_RANGE    = (0.0,   1.0)

N_PARAMS = 30   # 3 (L1) + 27 (L2)

PARAM_RANGES = [
    COEFF_RANGE if i % 3 == 0 else
    INTERCEPT_RANGE if i % 3 == 1 else
    HEIGHT_RANGE
    for i in range(N_PARAMS)
]


def denormalize(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        out[i] = lo + (np.clip(x[i], -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
    return out


# ─── Static data ──────────────────────────────────────────────────────────────

with open(OFF_FOOD_PATH) as f:
    OFF_FOOD = json.load(f)

# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _neutral_entry(coeff, intercept, height, p_off_food):
    return {
        "p_off_food":      p_off_food,
        "tau":             -1,
        "model_coeff":     coeff,
        "model_intercept": intercept,
        "model_height":    height,
        "mean":            0,
        "std":             1,
        "p_relevant":      0,
        "sign":            1,
    }


def write_l1(params: np.ndarray, l1_path: str):
    l1 = {}
    for idx, state in enumerate(L1_STATES):
        coeff      = float(params[idx * 3])
        intercept  = float(params[idx * 3 + 1])
        height     = float(params[idx * 3 + 2])
        p_off_food = OFF_FOOD[str(state)][str(state)]
        l1[str(state)] = _neutral_entry(coeff, intercept, height, p_off_food)
    with open(l1_path, "w") as f:
        json.dump(l1, f, indent=2)


def write_l2(params: np.ndarray, l2_path: str):
    l2 = {str(s): {} for s in STATES}
    for t_idx, (src, dst) in enumerate(TRANSITIONS):
        coeff      = float(params[3 + t_idx * 3])
        intercept  = float(params[3 + t_idx * 3 + 1])
        height     = float(params[3 + t_idx * 3 + 2])
        p_off_food = OFF_FOOD[str(src)][str(dst)]
        l2[str(src)][str(dst)] = _neutral_entry(coeff, intercept, height, p_off_food)
    with open(l2_path, "w") as f:
        json.dump(l2, f, indent=2)


# ─── Per-run simulator ────────────────────────────────────────────────────────

def run_single(cand_idx: int, run_idx: int, l1_path: str, l2_path: str) -> dict:
    """
    Launch one docker container for candidate cand_idx, ensemble run run_idx.
    Each run gets its own output directory and seed.
    Returns the parsed output dict.
    """
    out_dir = os.path.join(SIM_OUTPUT_DIR, f"cand{cand_idx:03d}_run{run_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # relative to container mount point /sim
    container_out_dir = f"/sim/outputs/cand{cand_idx:03d}_run{run_idx:02d}"
    container_l1      = f"/sim/outputs/cand{cand_idx:03d}_run{run_idx:02d}/l1.json"
    container_l2      = f"/sim/outputs/cand{cand_idx:03d}_run{run_idx:02d}/l2.json"

    # copy per-candidate JSONs into the run dir so each container reads its own
    shutil.copy(l1_path, os.path.join(out_dir, "l1.json"))
    shutil.copy(l2_path, os.path.join(out_dir, "l2.json"))

    seed = run_idx   # deterministic, distinct per ensemble run

    result = subprocess.run(
        [
            "bash", RUN_SCRIPT,
            "--output-dir", container_out_dir,
            "--l1",         container_l1,
            "--l2",         container_l2,
            "--seed",       str(seed),
        ],
        cwd=SIM_ROOT,
        capture_output=True,
        text=True,
    )

    #print("returncode:", result.returncode)
    #print("STDOUT:\n", result.stdout[-2000:])
    #print("STDERR:\n", result.stderr[-2000:])

    if result.returncode != 0:
        raise RuntimeError(
            f"[cand={cand_idx} run={run_idx}] docker failed:\n{result.stderr[-1000:]}"
        )

    out_path = os.path.join(out_dir, "auto_agents_100_all_data.json")
    with open(out_path) as f:
        return json.load(f)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_largest_cluster_fractions(pos: np.ndarray) -> np.ndarray:
    n_agents, n_frames, _ = pos.shape
    fractions = np.empty(n_frames)

    for t in range(n_frames):
        pts  = pos[:, t, :]
        diff = pts[:, None, :] - pts[None, :, :]
        diff[:, :, 0] -= DOMAIN_W * np.round(diff[:, :, 0] / DOMAIN_W)
        diff[:, :, 1] -= DOMAIN_H * np.round(diff[:, :, 1] / DOMAIN_H)
        dist = np.sqrt((diff ** 2).sum(axis=2))
        adj  = dist < NEIGHBOR_RADIUS_MM
        np.fill_diagonal(adj, False)

        parent = list(range(n_agents))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if adj[i, j]:
                    union(i, j)

        from collections import Counter
        sizes        = Counter(find(i) for i in range(n_agents))
        fractions[t] = max(sizes.values()) / n_agents

    return fractions


def compute_msd(pos: np.ndarray) -> np.ndarray:
    n_agents, n_frames, _ = pos.shape
    msd = np.zeros(n_frames)
    for tau in range(n_frames):
        disp = pos[:, tau, :] - pos[:, 0, :]
        disp[:, 0] -= DOMAIN_W * np.round(disp[:, 0] / DOMAIN_W)
        disp[:, 1] -= DOMAIN_H * np.round(disp[:, 1] / DOMAIN_H)
        msd[tau] = np.mean((disp ** 2).sum(axis=1))
    return msd


def compute_diffusion_coefficient(msd: np.ndarray) -> float:
    n  = len(msd)
    t  = np.arange(n, dtype=float)
    lo = max(1, int(0.1 * n))
    hi = int(0.9 * n)
    slope, _ = np.polyfit(t[lo:hi], msd[lo:hi], 1)
    return slope / 4.0


def metrics_from_data(data: dict) -> dict:
    avg_n              = float(data["avg_neighbors"])
    #pos                = np.array(data["positions"])    # (n_agents, n_frames, 2)
    #n_agents, n_frames, _ = pos.shape
    #com                = pos.mean(axis=0)
    #dist_to_com        = np.linalg.norm(pos - com[None, :, :], axis=2)
    #fractions          = compute_largest_cluster_fractions(pos)
    #msd                = compute_msd(pos)
    #diff_coeff         = compute_diffusion_coefficient(msd)
    return {
        "avg_neighbors":             avg_n#,
        #"mean_dist_to_com":          float(dist_to_com.mean()),
        #"largest_cluster_fractions": fractions,
        #"mean_cluster_size":         float(fractions.mean() * n_agents),
        #"diffusion_coefficient":     diff_coeff,
    }


# ─── Fitness functions ────────────────────────────────────────────────────────

def fitness_aggregation(metrics: dict) -> float:
    return -np.mean(metrics["avg_neighbors"])


def fitness_diffusion(metrics: dict) -> float:
    return np.mean(metrics["avg_neighbors"])


# ─── Parallel evaluate ────────────────────────────────────────────────────────

def evaluate_candidate(cand_idx: int, x_norm: np.ndarray) -> dict:
    """
    Run N_ENSEMBLE simulations in parallel for one candidate.
    Writes per-candidate l1/l2 JSONs to a temp dir.
    """
    params   = denormalize(x_norm)
    tmp_dir  = os.path.join(SIM_OUTPUT_DIR, f"params_cand{cand_idx:03d}")
    os.makedirs(tmp_dir, exist_ok=True)
    l1_path  = os.path.join(tmp_dir, "l1.json")
    l2_path  = os.path.join(tmp_dir, "l2.json")
    write_l1(params, l1_path)
    write_l2(params, l2_path)

    futures_data = []
    with ThreadPoolExecutor(max_workers=N_ENSEMBLE) as ex:
        futures = {
            ex.submit(run_single, cand_idx, run_idx, l1_path, l2_path): run_idx
            for run_idx in range(N_ENSEMBLE)
        }
        for fut in as_completed(futures):
            run_idx = futures[fut]
            try:
                futures_data.append(fut.result())
            except Exception as e:
                print(f"  [WARN] cand={cand_idx} run={run_idx} failed: {e}")

    if not futures_data:
        raise RuntimeError(f"All ensemble runs failed for candidate {cand_idx}")

    # average metrics across ensemble
    all_avg_n     = [m["avg_neighbors"]         for m in [metrics_from_data(d) for d in futures_data]]
    #all_dist_com  = [m["mean_dist_to_com"]       for m in [metrics_from_data(d) for d in futures_data]]
    #all_fractions = [m["largest_cluster_fractions"] for m in [metrics_from_data(d) for d in futures_data]]
    #all_diff      = [m["diffusion_coefficient"]  for m in [metrics_from_data(d) for d in futures_data]]

    #mean_fractions = np.mean(all_fractions, axis=0)

    return {
        "avg_neighbors":             float(np.mean(all_avg_n))#,
        #"mean_dist_to_com":          float(np.mean(all_dist_com)),
        #"largest_cluster_fractions": mean_fractions,
        #"mean_cluster_size":         float(mean_fractions.mean() * 100),
        #"diffusion_coefficient":     float(np.mean(all_diff)),
    }


def evaluate_population(solutions: list, fitness_fn) -> list:
    """
    Evaluate all candidates in parallel (candidates x ensemble simultaneously).
    """
    popsize = len(solutions)
    fitnesses = [1e6] * popsize

    with ThreadPoolExecutor(max_workers=popsize//2) as ex:
        futures = {
            ex.submit(evaluate_candidate, i, sol): i
            for i, sol in enumerate(solutions)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                metrics     = fut.result()
                fitnesses[i] = fitness_fn(metrics)
                print(f"  cand={i:02d} fit={fitnesses[i]:.4f}  "
                      f"avg_n={metrics['avg_neighbors']:.3f}  ")
                      #f"dist_com={metrics['mean_dist_to_com']:.3f}  "
                      #f"cluster={metrics['mean_cluster_size']:.3f}")
            except Exception as e:
                print(f"  cand={i:02d} ERROR ({e})")

    return fitnesses


# ─── CMA-ES ───────────────────────────────────────────────────────────────────

def run_cmaes(fitness_fn, label: str) -> tuple[np.ndarray, float]:
    x0 = np.zeros(N_PARAMS)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.6,
        {
            "maxiter": 200,
            "popsize": 14,   # 4 + floor(3 * ln(30))
            "verbose": 1,
            "tolx":    1e-11,
            "tolfun":  1e-11,
            "bounds":  [[-1.0] * N_PARAMS, [1.0] * N_PARAMS],
        },
    )

    print(f"\n{'='*60}")
    print(f"  Starting CMA-ES (parallelised): {label}")
    print(f"  candidates={es.popsize}  ensemble={N_ENSEMBLE}  "
          f"total_parallel={es.popsize * N_ENSEMBLE}")
    print(f"{'='*60}\n")

    best_params  = None
    best_fitness = np.inf
    iteration    = 0

    while not es.stop():
        solutions = es.ask()

        print(f"\n  [{label}] iter={iteration:03d} — "
              f"launching {len(solutions) * N_ENSEMBLE} containers...")

        fitnesses = evaluate_population(solutions, fitness_fn)

        es.tell(solutions, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        print(f"\n  [{label}] iter={iteration:03d} summary | "
              f"best_fit={fitnesses[best_idx]:.4f} | "
              f"mean_fit={np.mean(fitnesses):.4f} | "
              f"sigma={es.sigma:.4f}")

        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            best_params  = solutions[best_idx].copy()

        if iteration % LOG_EVERY == 0:
            _log_best(solutions[best_idx], fitnesses[best_idx], label, iteration)

        if es.stop():
            print(f"  [{label}] Stopped: {es.stop()}")

        iteration += 1

    print(f"\n[{label}] Done. Best fitness = {best_fitness:.4f}")
    return best_params, best_fitness


def _log_best(x_norm, fitness, label, iteration):
    """Re-run best candidate once (seed=0) and save cluster fractions."""
    params  = denormalize(x_norm)
    tmp_dir = os.path.join(SIM_OUTPUT_DIR, "log_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    l1_path = os.path.join(tmp_dir, "l1.json")
    l2_path = os.path.join(tmp_dir, "l2.json")
    write_l1(params, l1_path)
    write_l2(params, l2_path)

    try:
        data      = run_single(9999, 0, l1_path, l2_path)
        #pos       = np.array(data["positions"])
        #fractions = compute_largest_cluster_fractions(pos)
        log_path  = os.path.join(
            LOG_DIR, f"{label}_iter{iteration:04d}_cluster_fractions.json"
        )
        with open(log_path, "w") as f:
            json.dump({
                "iteration": iteration,
                "fitness":   fitness,
                #"fractions": fractions.tolist(),
            }, f, indent=2)
        print(f"  [LOG] Saved → {log_path}")
    except Exception as e:
        print(f"  [LOG] Failed to log iteration {iteration}: {e}")


# ─── Result saving ─────────────────────────────────────────────────────────────

def save_result(x_norm: np.ndarray, label: str):
    params = denormalize(x_norm)
    write_l1(params, L1_PATH)
    write_l2(params, L2_PATH)
    suffix = label.lower()
    for src, dst in [
        (L1_PATH, L1_PATH.replace(".json", f"_{suffix}.json")),
        (L2_PATH, L2_PATH.replace(".json", f"_{suffix}.json")),
    ]:
        with open(src) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)
    print(f"[{label}] Saved optimized JSONs with _{suffix} suffix.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(STATE_EST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="PFSM CMA-ES optimiser (parallelised)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a",  action="store_true", help="optimise aggregation")
    group.add_argument("-d",  action="store_true", help="optimise diffusion")
    group.add_argument("-ad", action="store_true", help="optimise both")
    args = parser.parse_args()

    if args.a or args.ad:
        agg_params, agg_fit = run_cmaes(fitness_aggregation, "AGGREGATION")
        save_result(agg_params, "aggregation")
        print(f"  Aggregation fitness : {agg_fit:.4f}")

    if args.d or args.ad:
        dif_params, dif_fit = run_cmaes(fitness_diffusion, "DIFFUSION")
        save_result(dif_params, "diffusion")
        print(f"  Diffusion fitness   : {dif_fit:.4f}")