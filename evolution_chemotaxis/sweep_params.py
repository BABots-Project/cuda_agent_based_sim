"""
sweep_all_pairs.py
------------------
Runs grid_sweep.py for every pair of parameters sequentially.
Results land in sweep_<p1>_vs_<p2>.json + .png for each pair.

Usage
-----
    python sweep_all_pairs.py
    python sweep_all_pairs.py --base best_params.json --n 5 --out-dir sweeps/
"""

import argparse
import logging
import subprocess
import sys
from itertools import combinations
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PARAM_NAMES = [
    "p1_minus",
    "a1",
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


def parse_args():
    p = argparse.ArgumentParser(description="Sweep all parameter pairs")
    p.add_argument("--n",       type=int, default=5,   help="Grid points per axis")
    p.add_argument("--base",    default=None,           help="Base params JSON")
    p.add_argument("--out-dir", default=".",            help="Directory for output files")
    p.add_argument("--skip-zeros", action="store_true",
                   help="Skip pairs where both parameters are fixed to 0 in base params")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(combinations(PARAM_NAMES, 2))
    total = len(pairs)
    log.info("Total pairs: %d  (%d evals each → %d total simulations)",
             total, args.n ** 2, total * args.n ** 2)

    # optionally load base params to skip all-zero pairs
    base_params = {}
    if args.base:
        import json
        with open(args.base) as f:
            base_params = json.load(f)

    skipped = []
    if args.skip_zeros:
        pairs_filtered = []
        for p1, p2 in pairs:
            if base_params.get(p1, 1.0) == 0.0 and base_params.get(p2, 1.0) == 0.0:
                log.info("Skipping %s vs %s (both fixed to 0)", p1, p2)
                skipped.append((p1, p2))
            else:
                pairs_filtered.append((p1, p2))
        pairs = pairs_filtered
        log.info("Skipped %d pairs, running %d", len(skipped), len(pairs))

    failed = []
    for idx, (p1, p2) in enumerate(pairs):
        out_path = out_dir / f"sweep_{p1}_vs_{p2}.png"

        # skip if already done
        if out_path.exists():
            log.info("[%d/%d] Skipping %s vs %s (already exists)",
                     idx + 1, len(pairs), p1, p2)
            continue

        log.info("[%d/%d] Sweeping %s vs %s", idx + 1, len(pairs), p1, p2)

        cmd = [
            sys.executable, "grid_sweep.py",
            "--p1", p1,
            "--p2", p2,
            "--n",  str(args.n),
            "--out", str(out_path),
        ]
        if args.base:
            cmd += ["--base", args.base]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            log.warning("  FAILED: %s vs %s (exit code %d)", p1, p2, result.returncode)
            failed.append((p1, p2))
        else:
            log.info("  Done → %s", out_path)

    log.info("All pairs complete.")
    if failed:
        log.warning("%d pairs failed: %s", len(failed), failed)
    else:
        log.info("No failures.")


if __name__ == "__main__":
    main()