"""
optimize_params.py
------------------
Optimises the 8 agent parameters using Differential Evolution.

Fitness = Wasserstein distance between real and simulated distributions of
    - fraction of frames with dC/dt > 0  *before*  the odour hit
    - fraction of frames with dC/dt > 0  *after*   the odour hit

The simulation is run by calling ./run_sim.sh (params.json is hardcoded in that
script).  This optimizer writes params.json before each call and reads the
output from the hardcoded output JSON path.

Usage
-----
    python optimize_params.py [--help]

Adjust the constants at the top of the file to match your paths / settings.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import wasserstein_distance

# ── user-facing constants ─────────────────────────────────────────────────────

SIM_SCRIPT       = "./offline_build_and_run.sh"          # bash script that runs the sim
PARAMS_JSON      = "../state_estimations/chemotaxis_params.json"           # path the sim reads params from
SIM_OUTPUT_JSON  = "../auto_agents_100_all_data.json"

# real-worm reference distributions (pre-computed once; see compute_real_stats)
# Set REAL_STATS_CACHE to a file path to cache them across runs.
REAL_STATS_CACHE = "real_worm_stats.json"

# Number of independent seeds per candidate evaluation (averaged for fitness)
N_SEEDS = 1

# Timeout per single simulation run (seconds)
SIM_TIMEOUT = 300

# Differential Evolution settings
DE_POPSIZE   = 5     # population = popsize * n_params
DE_MAXITER   = 150
DE_TOL       = 1e-4
DE_MUTATION  = (0.5, 1.0)
DE_RECOMB    = 0.7
DE_WORKERS   = 1     # set to -1 to use all CPU cores (each worker runs N_SEEDS sims)
DE_SEED      = 42

# Weight of "before hit" vs "after hit" Wasserstein distances in the total loss
W_BEFORE = 0.5
W_AFTER  = 0.5

# Odor hit distance threshold (mm)
HIT_DIST_MM = 5.0

# diffusion profile parameters forwarded to compute_C_trace
T_START = 10.0
DT      = 1 / 3

# ── parameter layout ──────────────────────────────────────────────────────────

PARAM_NAMES = [
    "p_run_rev_minus",
    "p_run_turn_minus",
    "p_rev_run_minus",
    "p_rev_turn_minus",
    "p_turn_run_minus",
    "p_turn_rev_minus",
    "a_run_rev",
    "a_run_turn",
    "a_rev_run",
    "a_rev_turn",
    "a_turn_run",
    "a_turn_rev",
]

# All parameters are probabilities in [0, 1]
BOUNDS = [(0.0, 1.0)] * len(PARAM_NAMES)

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── diffusion helpers (keep in sync with your utils) ─────────────────────────

def diffusion_profile_distance(r: float, t: float,
                               D: float = 1.0, Q: float = 1.0) -> float:
    """Gaussian diffusion concentration at distance r at time t."""
    if t <= 0:
        return 0.0
    return Q / (4 * np.pi * D * t) * np.exp(-(r ** 2) / (4 * D * t))


def compute_C_trace(xy_mm: np.ndarray, odor_orig: tuple,
                    t_start: float = T_START, dt: float = DT):
    """Return (C, dC, distance) for a single trajectory."""
    distance = np.linalg.norm(xy_mm - np.array(odor_orig), axis=1)
    C = np.array([
        diffusion_profile_distance(distance[i], t_start + i * dt)
        for i in range(len(xy_mm))
    ])
    dC = np.diff(C, prepend=C[0])
    return C, dC, distance


# ── extract stats from a loaded JSON (sim or real) ───────────────────────────

def extract_hit_stats(data: dict) -> tuple[list, list]:
    """
    Given a loaded simulation JSON dict, return
        (before_hit_fractions, after_hit_fractions)
    for every agent that reaches within HIT_DIST_MM of the odour.
    """
    positions = np.array(data["positions"])       # (n_agents, T, 2)
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


# ── real-worm reference ───────────────────────────────────────────────────────

def load_real_stats(real_data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (or compute + cache) the real-worm reference distributions.
    `real_data_path` should be a JSON in the same format as the sim output.
    """
    cache = Path(REAL_STATS_CACHE)
    if cache.exists():
        log.info("Loading real-worm stats from cache: %s", cache)
        with open(cache) as f:
            d = json.load(f)
        return np.array(d["before"]), np.array(d["after"])

    log.info("Computing real-worm stats from %s …", real_data_path)
    with open(real_data_path) as f:
        data = json.load(f)
    before, after = extract_hit_stats(data)

    with open(cache, "w") as f:
        json.dump({"before": before, "after": after}, f)
    log.info("Cached real-worm stats to %s  (before n=%d, after n=%d)",
             cache, len(before), len(after))
    return np.array(before), np.array(after)


# ── simulation runner ─────────────────────────────────────────────────────────

def write_params(params_dict: dict, path: str = PARAMS_JSON) -> None:
    with open(path, "w") as f:
        json.dump(params_dict, f, indent=2)

SIM_DIR = Path("~/cuda_agent_based_sim").expanduser()
#SIM_DIR = Path("../cuda_agent_based_sim")  # adjust ../ count as needed
#SIM_DIR = Path(__file__).resolve().parent #/ "../cuda_agent_based_sim"
#SIM_DIR = SIM_DIR.resolve()
SIM_SCRIPT = SIM_DIR / "offline_build_and_run.sh"
def run_sim(seed: int) -> dict | None:
    """Write params, call the bash script with the given seed, return JSON."""
    # Optionally pass seed as an env var so the sim can pick it up
    import os
    env = os.environ.copy()
    env["SIM_SEED"] = str(seed)

    try:
        print(f"running {str(SIM_SCRIPT)}")
        result = subprocess.run(
            [str(SIM_SCRIPT)],   # absolute path, no ambiguity
            env=env,
            cwd=SIM_DIR,
            timeout=SIM_TIMEOUT,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("Simulation timed out (seed=%d)", seed)
        return None
    except Exception as e:
        log.warning("Simulation failed (seed=%d): %s", seed, e)
        return None

    if result.returncode != 0:
        log.warning("Simulation non-zero exit (seed=%d):\n%s", seed, result.stderr[-500:])
        return None

    try:
        with open(SIM_OUTPUT_JSON) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not read sim output (seed=%d): %s", seed, e)
        return None


# ── fitness function ──────────────────────────────────────────────────────────

_eval_counter = 0   # global counter for logging

def fitness(x: np.ndarray,
            real_before: np.ndarray,
            real_after: np.ndarray) -> float:
    """
    Evaluate a parameter vector x (length 8).
    Runs N_SEEDS simulations and returns mean Wasserstein fitness.
    Lower is better (DE minimises by default).
    """
    global _eval_counter
    _eval_counter += 1

    params_dict = dict(zip(PARAM_NAMES, x.tolist()))
    write_params(params_dict)
    log.info("[eval %d] params: %s",
             _eval_counter,
             "  ".join(f"{k}={v:.4f}" for k, v in params_dict.items()))

    seed_scores = []
    for seed in range(N_SEEDS):
        data = run_sim(seed)
        if data is None:
            log.warning("  seed %d failed — skipping", seed)
            continue

        sim_before, sim_after = extract_hit_stats(data)

        if len(sim_before) < 2 or len(sim_after) < 2:
                    log.warning("  seed %d: too few hitting agents (%d before, %d after)",
                                seed, len(sim_before), len(sim_after))
                    # penalise — no agents reached the odour
                    seed_scores.append(1.0)
                    continue

        w_before = wasserstein_distance(real_before, sim_before)
        w_after  = wasserstein_distance(real_after,  sim_after)
        score    = W_BEFORE * w_before + W_AFTER * w_after
        seed_scores.append(score)
        log.info("  seed %d: W_before=%.4f  W_after=%.4f  score=%.4f",
                 seed, w_before, w_after, score)

    if not seed_scores:
        log.warning("  all seeds failed — returning penalty 1.0")
        return 1.0

    mean_score = float(np.mean(seed_scores))
    log.info("[eval %d] mean fitness = %.4f", _eval_counter, mean_score)
    return mean_score


# ── callback (progress logging) ───────────────────────────────────────────────

_best_so_far = np.inf

def de_callback(xk, convergence):
    """Called after each generation."""
    global _best_so_far
    # scipy doesn't pass the current best directly; we track it ourselves
    log.info("── generation done  convergence=%.6f  best_so_far=%.4f",
             convergence, _best_so_far)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DE optimiser for agent sim parameters")
    p.add_argument("--real-data", default=SIM_OUTPUT_JSON,
                   help="Path to real-worm JSON (default: SIM_OUTPUT_JSON)")
    p.add_argument("--popsize",  type=int,   default=DE_POPSIZE)
    p.add_argument("--maxiter",  type=int,   default=DE_MAXITER)
    p.add_argument("--n-seeds",  type=int,   default=N_SEEDS)
    p.add_argument("--workers",  type=int,   default=DE_WORKERS,
                   help="-1 = all cores")
    p.add_argument("--output",   default="best_params.json",
                   help="Where to save the best parameters found")
    return p.parse_args()

def build_init_population(x0, popsize, n_params, bounds, spread=0.1):
    """x0: your hand-found values (in PARAM_NAMES order). Returns init population array."""
    pop_size = popsize * n_params
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, n_params),
    )
    # perturb individuals around x0 so the optimizer starts near your known-good point
    population[0] = x0
    for i in range(1, pop_size // 2):
        noise = np.random.normal(0, spread, size=n_params)
        population[i] = np.clip(x0 + noise, [b[0] for b in bounds], [b[1] for b in bounds])
    return population


def main():
    args = parse_args()

    global N_SEEDS
    N_SEEDS = args.n_seeds

    # 1. Load reference distributions
    real_before, real_after = load_real_stats(args.real_data)
    log.info("Real-worm stats loaded  (before n=%d, after n=%d)",
             len(real_before), len(real_after))

    # 2. Wrap fitness so DE can call it with just x
    def objective(x):
        global _best_so_far
        score = fitness(x, real_before, real_after)
        if score < _best_so_far:
            _best_so_far = score
            # Save intermediate best immediately
            best = dict(zip(PARAM_NAMES, x.tolist()))
            best["fitness"] = score
            with open(args.output, "w") as f:
                json.dump(best, f, indent=2)
            log.info("★ New best saved → %s (fitness=%.4f)", args.output, score)
        return score

    x0 = np.array([
        0.4878576636846718,
        0.4,
        0.2,
        0.6,
        0.2,
        0.4,
        0.1,
        0.1,
        0.8,
        0.1,
        0.8,
        0.1
    ])

    # 3. Run Differential Evolution
    log.info("Starting Differential Evolution  popsize=%d  maxiter=%d  workers=%d",
             args.popsize, args.maxiter, args.workers)
    t0 = time.time()

    init_pop = build_init_population(x0, args.popsize, len(PARAM_NAMES), BOUNDS)

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        init=init_pop,        # <-- replaces popsize-driven default init
        maxiter=args.maxiter,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMB,
        seed=DE_SEED,
        workers=args.workers,
        callback=de_callback,
        polish=True,
        disp=True,
    )

    elapsed = time.time() - t0
    log.info("Optimisation finished in %.1f s", elapsed)
    log.info("Success: %s", result.success)
    log.info("Message: %s", result.message)
    log.info("Best fitness: %.6f", result.fun)

    best_params = dict(zip(PARAM_NAMES, result.x.tolist()))
    best_params["fitness"] = float(result.fun)
    best_params["n_evals"] = _eval_counter
    best_params["elapsed_s"] = elapsed

    with open(args.output, "w") as f:
        json.dump(best_params, f, indent=2)

    log.info("Best parameters saved to %s", args.output)
    for k, v in best_params.items():
        log.info("  %-24s %s", k, v)


if __name__ == "__main__":
    main()