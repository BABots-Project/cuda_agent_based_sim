import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
PERM          = "000"
METHODS       = ["KMeans", "HDBSCAN", "Ward", "Spectral"]
SIM_RESULTS   = Path("../sim_results")
RESULTS_DIR   = Path("../results")
PERMS_BASE    = Path("../state_estimations/permutations")
CI_LO, CI_HI  = 5, 95

COLORS = {
    "KMeans"  : "#2a78d6",
    "HDBSCAN" : "#1baf7a",
    "Ward"    : "#eda100",
    "Spectral": "#4a3aa7",
}
BIO_COLOR = "#e34948"

# ── LOAD PRECOMPUTED BIO MSDs ─────────────────────────────────────────────────
bio_msds_npz = np.load(RESULTS_DIR / "off_food_biological_msds.npz")

# test worms for this permutation
with open(PERMS_BASE / f"perm_{PERM}" / "split.json") as f:
    split = json.load(f)
test_ids = np.array(split["test"])

bio_msd_curves = [bio_msds_npz["worm_"+str(g)] for g in test_ids]

max_lag = min(len(m) for m in bio_msd_curves)
lags    = np.arange(1, max_lag + 1)

bio_arr = np.array([m[:max_lag] for m in bio_msd_curves])
bio_mid = np.median(bio_arr, axis=0)
bio_lo  = np.min(bio_arr,    axis=0)
bio_hi  = np.max(bio_arr,    axis=0)

# ── MSD ───────────────────────────────────────────────────────────────────────
def compute_msd(positions, max_lag):
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]
        msd[lag - 1]  = np.mean(np.sum(displacements ** 2, axis=1))
    return msd

# ── SIM LOADER ────────────────────────────────────────────────────────────────
def load_sim_msds(perm, method, agent_ids, max_lag):
    msds = []
    for aid in agent_ids:
        p = SIM_RESULTS / f"perm_{perm}" / method / f"simulated_worm_{aid}.json"
        with open(p) as f:
            data = json.load(f)
        print(f"available keys {data.keys()}")
        positions = data["positions"] if isinstance(data, dict) else data
        states = data["sub_states"]
        speed = data["velocities"]
        angle_changes = data["angles"]
        state_speeds = {0:[], 1:[], 2:[]}
        state_acs = {0:[], 1:[], 2:[]}
        for i in range(len(positions)):
            agent_labels = np.array(states[i])
            agent_speeds = np.array(speed[i])
            agent_angle_changes = np.array(angle_changes[i])
            for s in range(3):
                state_idx = agent_labels == s
                state_speed = agent_speeds[state_idx]
                state_angle_changes = agent_angle_changes[state_idx]
                state_speeds[s].extend(state_speed.tolist())
                state_acs[s].extend(state_angle_changes.tolist())
        '''for i in range(3):
            plt.subplots(1,2, figsize=(12,6))
            plt.subplot(1,2,1)
            plt.hist(state_speeds[i], bins=30)
            plt.title(f"speed state {i}")
            plt.subplot(1,2,2)
            plt.hist(state_acs[i], bins=30)
            plt.title(f"angle change state {i}")
            plt.show()'''
        for agent_pos in positions:
            pos = np.array(agent_pos)
            msds.append(compute_msd(pos, max_lag))

    return np.array(msds)   # (n_runs, max_lag)

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"MSD — biological vs simulated  |  perm {PERM}  |  {CI_LO}–{CI_HI}% CI",
    fontsize=13, y=1.01
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
sim_arrs = []
for ax_idx, method in enumerate(METHODS):
    ax  = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2])
    col = COLORS[method]

    # bio envelope
    ax.fill_between(lags, bio_lo, bio_hi,
                    color=BIO_COLOR, alpha=0.25, linewidth=0, zorder=1)
    ax.plot(lags, bio_mid, color=BIO_COLOR, lw=2.0,
            label="biological (median, min–max)", zorder=3)

    # simulated envelope
    try:
        sim_arr = load_sim_msds(PERM, method, test_ids, max_lag)
        sim_arrs.append(sim_arr)
        sim_mid = np.median(sim_arr,            axis=0)
        sim_lo  = np.percentile(sim_arr, CI_LO, axis=0)
        sim_hi  = np.percentile(sim_arr, CI_HI, axis=0)

        ax.fill_between(lags, sim_lo, sim_hi,
                        color=col, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(lags, sim_mid, color=col, lw=2.0,
                label=f"{method} (median, {CI_LO}–{CI_HI}% CI)", zorder=3)
        sim_loaded = True
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, f"missing:\n{Path(e.filename).name}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="gray")
        sim_loaded = False

    ax.set_xlim(1, max_lag)
    ax.set_xlabel("Lag (frames)", fontsize=11)
    ax.set_ylabel("MSD (mm²)", fontsize=11)
    ax.set_title(method, fontsize=11, fontweight="normal")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

out_stem = f"msd_comparison_perm{PERM}"
fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved {out_stem}.pdf / .png")

# ── compare simulated MSD envelopes ───────────────────────────────────────

method_msds = {}

for i, method in enumerate(METHODS):
    sim_arr = sim_arrs[i]
    method_msds[method] = {
        "median": np.median(sim_arr, axis=0),
        "lo": np.percentile(sim_arr, CI_LO, axis=0),
        "hi": np.percentile(sim_arr, CI_HI, axis=0),
    }

# pairwise normalized differences between median MSD curves
for i, m1 in enumerate(METHODS):
    for m2 in METHODS[i+1:]:
        a = method_msds[m1]["median"]
        b = method_msds[m2]["median"]

        nrmse = np.sqrt(np.mean((a-b)**2)) / np.mean((a+b)/2)
        corr = np.corrcoef(a, b)[0,1]

        print(
            f"{m1:10s} vs {m2:10s} | "
            f"NRMSE={nrmse:.4f} | corr={corr:.4f}"
        )

# envelope overlap
for i, m1 in enumerate(METHODS):
    for m2 in METHODS[i+1:]:
        lo = np.maximum(method_msds[m1]["lo"], method_msds[m2]["lo"])
        hi = np.minimum(method_msds[m1]["hi"], method_msds[m2]["hi"])

        overlap = np.mean(np.maximum(0, hi-lo) /
                          (method_msds[m1]["hi"]-method_msds[m1]["lo"]))

        print(f"{m1:10s} vs {m2:10s} envelope overlap={overlap:.3f}")