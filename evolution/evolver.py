"""
CMA-ES optimization of PFSM logistic parameters (L1 and L2).

Parameter vector layout (24 values total, normalized to [-1, 1]):
  [0:6]   L1 - per state (3 states x 2 params: coeff, intercept)
  [6:24]  L2 - per transition (9 transitions x 2 params: coeff, intercept)

Transitions order: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)

Fitness:
  Aggregation : minimize  mean_dist_to_com - avg_neighbors
  Diffusion   : minimize  avg_neighbors + 1 / (mean_dist_to_com + eps)
"""

import json
import subprocess
import os
import numpy as np
import cma

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATE_EST_DIR  = os.path.join(SIM_ROOT, "state_estimations")
SIM_OUTPUT_DIR = SIM_ROOT
RUN_SCRIPT     = os.path.join(SIM_ROOT, "offline_build_and_run.sh")

L1_PATH       = os.path.join(STATE_EST_DIR, "l1.json")
L2_PATH       = os.path.join(STATE_EST_DIR, "l2.json")
OFF_FOOD_PATH = os.path.join(STATE_EST_DIR, "off_food_transitions.json")

AGENT_IDS   = list(range(37, 46))
STATES      = [0, 1, 2]
TRANSITIONS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

NEIGHBOR_RADIUS_MM = 0.5
EPS                = 1e-6

# ─── Parameter scaling ────────────────────────────────────────────────────────

COEFF_RANGE     = (-5.0,  5.0)
INTERCEPT_RANGE = (-20.0, 20.0)  # crossing always in valid N range
HEIGHT_RANGE    = (0.0,   1.0)

'''PARAM_RANGES = [
    COEFF_RANGE if i % 2 == 0 else INTERCEPT_RANGE
    for i in range(24)
]'''
# triplets: coeff, intercept, height — 1 for L1, 9 for L2
PARAM_RANGES = [
    COEFF_RANGE if i % 3 == 0 else
    INTERCEPT_RANGE if i % 3 == 1 else
    HEIGHT_RANGE
    for i in range(30)  # 3 + 27
]

DOMAIN_W = 60.0
DOMAIN_H = 60.0

LOG_EVERY   = 20
LOG_DIR     = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
L1_STATES = [2]  # only evolve L1 for state 2

def denormalize(x: np.ndarray) -> np.ndarray:
    """Map normalized [-1, 1] vector to actual parameter ranges."""
    out = np.empty_like(x)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        out[i] = lo + (np.clip(x[i], -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
    return out


# ─── Static data ──────────────────────────────────────────────────────────────

with open(OFF_FOOD_PATH) as f:
    OFF_FOOD = json.load(f)

# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _neutral_entry(coeff: float, intercept: float, height:float, p_off_food: float) -> dict:
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


def write_l1(params: np.ndarray):
    l1 = {}
    for idx, state in enumerate(L1_STATES):
        coeff      = float(params[idx * 3])
        intercept  = float(params[idx * 3 + 1])
        height     = float(params[idx * 3 + 2])
        p_off_food = OFF_FOOD[str(state)][str(state)]
        l1[str(state)] = _neutral_entry(coeff, intercept, height, p_off_food)
    with open(L1_PATH, "w") as f:
        json.dump(l1, f, indent=2)


def write_l2(params: np.ndarray):
    l2 = {str(s): {} for s in STATES}
    for t_idx, (src, dst) in enumerate(TRANSITIONS):
        coeff      = float(params[3 + t_idx * 3])
        intercept  = float(params[3 + t_idx * 3 + 1])
        height     = float(params[3 + t_idx * 3 + 2])
        p_off_food = OFF_FOOD[str(src)][str(dst)]
        l2[str(src)][str(dst)] = _neutral_entry(coeff, intercept, height, p_off_food)
    with open(L2_PATH, "w") as f:
        json.dump(l2, f, indent=2)


# ─── Simulator interface ───────────────────────────────────────────────────────

def run_simulator():
    result = subprocess.run(
        ["bash", RUN_SCRIPT],
        cwd=SIM_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[SIM STDERR]", result.stderr[-2000:])
        raise RuntimeError(f"Simulator exited with code {result.returncode}")


def load_output() -> dict:
    path = os.path.join(SIM_OUTPUT_DIR, f"auto_agents_100_all_data.json")
    with open(path) as f:
        return json.load(f)


def compute_metrics(agent_id: int) -> dict:
    """
    Returns:
      avg_neighbors      : float  — read directly from sim output
      mean_dist_to_com   : float  — mean over frames of each agent's dist to CoM
      cluster_sizes      : list   — distribution of cluster sizes (last frame only)
    """
    data   = load_output(agent_id)
    avg_n  = float(data["avg_neighbors"])

    # positions: [n_agents][n_frames][2]
    pos = np.array(data["positions"])   # (n_agents, n_frames, 2)
    n_agents, n_frames, _ = pos.shape

    # ── Mean distance to centre of mass ──────────────────────────────────────
    # CoM per frame: (n_frames, 2)
    com = pos.mean(axis=0)                          # (n_frames, 2)
    # distance of each agent to CoM per frame: (n_agents, n_frames)
    dist_to_com = np.linalg.norm(pos - com[None, :, :], axis=2)
    mean_dist_to_com = float(dist_to_com.mean())

    # ── Cluster size distribution (last frame) ────────────────────────────────
    last = pos[:, -1, :]                            # (n_agents, 2)
    # adjacency: agents within NEIGHBOR_RADIUS_MM
    diff = last[:, None, :] - last[None, :, :]     # (n, n, 2)
    adj  = np.sqrt((diff ** 2).sum(axis=2)) < NEIGHBOR_RADIUS_MM
    np.fill_diagonal(adj, False)

    # connected components via union-find
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
    roots        = [find(i) for i in range(n_agents)]
    cluster_sizes = list(Counter(roots).values())

    return {
        "avg_neighbors":    avg_n,
        "mean_dist_to_com": mean_dist_to_com,
        "cluster_sizes":    cluster_sizes,
    }


N_ENSEMBLE = 5

def evaluate(x_norm: np.ndarray) -> dict:
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)

    all_avg_n       = []
    all_dist_com    = []
    all_fractions   = []

    for run in range(N_ENSEMBLE):
        run_simulator()

        path = os.path.join(SIM_OUTPUT_DIR, "auto_agents_100_all_data.json")
        with open(path) as f:
            data = json.load(f)

        avg_n         = float(data["avg_neighbors"])
        pos           = np.array(data["positions"])   # (n_agents, n_frames, 2)
        n_agents, n_frames, _ = pos.shape

        com           = pos.mean(axis=0)
        dist_to_com   = np.linalg.norm(pos - com[None, :, :], axis=2)
        fractions     = compute_largest_cluster_fractions(pos)

        all_avg_n.append(avg_n)
        all_dist_com.append(float(dist_to_com.mean()))
        all_fractions.append(fractions)

    # average metrics across ensemble
    mean_fractions = np.mean(all_fractions, axis=0)  # (n_frames,)

    return {
        "avg_neighbors":             float(np.mean(all_avg_n)),
        "mean_dist_to_com":          float(np.mean(all_dist_com)),
        "mean_cluster_size":         float(np.mean(mean_fractions) * n_agents),
        "largest_cluster_fractions": mean_fractions,
    }




def compute_largest_cluster_fractions(pos: np.ndarray) -> np.ndarray:
    n_agents, n_frames, _ = pos.shape
    fractions = np.empty(n_frames)

    for t in range(n_frames):
        pts  = pos[:, t, :]                         # (n_agents, 2)
        diff = pts[:, None, :] - pts[None, :, :]    # (n, n, 2)

        # wrap for torus
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
        sizes     = Counter(find(i) for i in range(n_agents))
        fractions[t] = max(sizes.values()) / n_agents

    return fractions

def compute_msd(pos: np.ndarray) -> np.ndarray:
    """
    pos: (n_agents, n_frames, 2)
    Returns MSD at each time lag from 0 to n_frames-1.
    """
    n_agents, n_frames, _ = pos.shape
    msd = np.zeros(n_frames)
    for tau in range(n_frames):
        # displacement from initial position at lag tau
        disp = pos[:, tau, :] - pos[:, 0, :]
        # wrap for periodic boundaries
        disp[:, 0] -= DOMAIN_W * np.round(disp[:, 0] / DOMAIN_W)
        disp[:, 1] -= DOMAIN_H * np.round(disp[:, 1] / DOMAIN_H)
        msd[tau] = np.mean((disp ** 2).sum(axis=1))
    return msd


def compute_diffusion_coefficient(msd: np.ndarray) -> float:
    """Fit MSD = 4Dt (2D diffusion) and return D."""
    n = len(msd)
    t = np.arange(n, dtype=float)
    # fit only the linear portion — skip first 10% (ballistic) and last 10% (boundary effects)
    lo = max(1, int(0.1 * n))
    hi = int(0.9 * n)
    slope, _ = np.polyfit(t[lo:hi], msd[lo:hi], 1)
    return slope / 4.0  # D = slope / (2 * n_dims)

def fitness_aggregation(metrics: dict) -> float:
    #return -np.mean(metrics["largest_cluster_fractions"])
    return -metrics["avg_neighbors"] #

def fitness_diffusion(metrics: dict) -> float:
    # minimize neighbors, maximize spread → minimize avg_n + 1/(dist_to_com + eps)
    #return metrics["avg_neighbors"] + 1.0 / (metrics["mean_dist_to_com"] + EPS)
    #return -metrics["diffusion_coefficient"]  # maximize D
    return metrics["avg_neighbors"]
# ─── CMA-ES ───────────────────────────────────────────────────────────────────

def run_cmaes(fitness_fn, label: str) -> tuple[np.ndarray, float]:
    x0 = np.zeros(30)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.6,            # sigma in normalized space — 0.5 is a quarter of [-1,1]
        {
            "maxiter": 200,
            "popsize": 14,  # 4 + floor(3 * ln(24))
            "verbose": 1,
            "tolx":    1e-6,
            "tolfun":  1e-6,
            "bounds": [[-1.0] * 30, [1.0] * 30],
        },
    )

    print(f"\n{'='*60}")
    print(f"  Starting CMA-ES: {label}")
    print(f"{'='*60}\n")

    best_params = None
    best_fitness = np.inf
    iteration = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for i, sol in enumerate(solutions):
            print(f"  [{label}] iter={iteration:03d} cand={i:02d} ...", end=" ", flush=True)
            try:
                metrics = evaluate(sol)
                fit     = fitness_fn(metrics)
                print(f"fit={fit:.4f}  avg_n={metrics['avg_neighbors']:.3f}  "
                      f"dist_com={metrics['mean_dist_to_com']:.3f}  "
                      f"cluster={metrics['mean_cluster_size']:.3f}")
            except Exception as e:
                fit = 1e6
                print(f"ERROR ({e})")

            fitnesses.append(fit)

            if fit < best_fitness:
                best_fitness = fit
                best_params  = sol.copy()

        es.tell(solutions, fitnesses)
        best_idx = int(np.argmin(fitnesses))
        print(f"\n  [{label}] iter={iteration:03d} summary | "
              f"best_fit={fitnesses[best_idx]:.4f} | "
              f"mean_fit={np.mean(fitnesses):.4f} | "
              f"sigma={es.sigma:.4f}")

        if iteration % LOG_EVERY == 0:
            # re-evaluate best candidate of this iteration to get the full fractions
            best_sol = solutions[best_idx]
            params   = denormalize(best_sol)
            write_l1(params)
            write_l2(params)
            run_simulator()

            path = os.path.join(SIM_OUTPUT_DIR, "auto_agents_100_all_data.json")
            with open(path) as f:
                data = json.load(f)
            pos       = np.array(data["positions"])
            fractions = compute_largest_cluster_fractions(pos)

            log_path = os.path.join(LOG_DIR, f"{label}_iter{iteration:04d}_cluster_fractions.json")
            with open(log_path, "w") as f:
                json.dump({
                    "iteration":  iteration,
                    "fitness":    fitnesses[best_idx],
                    "fractions":  fractions.tolist(),   # one value per timestep
                }, f, indent=2)
            print(f"  [LOG] Saved cluster fractions → {log_path}")

        iteration += 1
    print(f"  [{label}] Stopped because: {es.stop()}")
    print(f"\n[{label}] Done. Best fitness = {best_fitness:.4f}")
    return best_params, best_fitness


# ─── Result saving ─────────────────────────────────────────────────────────────

def save_result(x_norm: np.ndarray, label: str):
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)
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
def write_empty_l1l2():
    """Write l1/l2 with all model params set to -1 (baseline/control)."""
    l1 = {}
    for state in STATES:
        p_off_food = OFF_FOOD[str(state)][str(state)]
        l1[str(state)] = _neutral_entry(-1.0, -1.0, -1.0, p_off_food)
    with open(L1_PATH, "w") as f:
        json.dump(l1, f, indent=2)

    l2 = {str(s): {} for s in STATES}
    for src, dst in TRANSITIONS:
        p_off_food = OFF_FOOD[str(src)][str(dst)]
        l2[str(src)][str(dst)] = _neutral_entry(-1.0, -1.0, -1.0, p_off_food)
    with open(L2_PATH, "w") as f:
        json.dump(l2, f, indent=2)

# ─── Entry point ──────────────────────────────────────────────────────────────

import argparse

if __name__ == "__main__":
    os.makedirs(STATE_EST_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="PFSM CMA-ES optimizer")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a",  action="store_true", help="optimize aggregation")
    group.add_argument("-d",  action="store_true", help="optimize diffusion")
    group.add_argument("-ad", action="store_true", help="optimize both")
    args = parser.parse_args()

    if args.a or args.ad:
        agg_params, agg_fit = run_cmaes(fitness_aggregation, "AGGREGATION")
        save_result(agg_params, "aggregation")
        print(f"  Aggregation fitness : {agg_fit:.4f}")

    if args.d or args.ad:
        dif_params, dif_fit = run_cmaes(fitness_diffusion, "DIFFUSION")
        save_result(dif_params, "diffusion")
        print(f"  Diffusion fitness   : {dif_fit:.4f}")