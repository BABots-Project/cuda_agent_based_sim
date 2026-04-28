from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
import math

def plot_feature_distributions_by_cluster(
        data,
        labels,
        feature_names,
        selected_feature_ids=None,
        bins=30,
        max_cols=4,
        figsize_per_subplot=(4, 3),
        cmap_name='tab10',
        exclude_noise=True,
        show=True,
        save_dir=None
):
    """
    Plot per-feature distributions separated by cluster.

    Parameters
    ----------
    data : np.ndarray shape (N, F)
        Data matrix (e.g. normalised_data).
    labels : array-like shape (N,)
        Cluster labels (HDBSCAN labels). Noise label -1 will be excluded if exclude_noise True.
    feature_names : list[str] length F
        Names for the F features (in same order as columns in `data`).
    selected_feature_ids : list[int] or None
        Indices of features to plot. If None, will plot all columns in `data`.
    bins : int
        Number of bins for the histograms.
    max_cols : int
        Maximum number of subplot columns per figure.
    figsize_per_subplot : tuple (w,h)
        Size per subplot; final figure size = cols*w by rows*h.
    cmap_name : str
        Matplotlib colormap name for cluster colors.
    exclude_noise : bool
        If True, exclude label -1.
    show : bool
        If True, call plt.show() for each figure.
    save_dir : str or None
        If provided, each figure will be saved as {save_dir}/{feature_name}.png

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        The created Figure objects (one per feature plotted).
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    data = np.asarray(data)
    labels = np.asarray(labels)

    if selected_feature_ids is None:
        selected_feature_ids = list(range(data.shape[1]))

    # cluster labels excluding noise if requested
    all_labels = np.unique(labels)
    if exclude_noise and (-1 in all_labels):
        cluster_labels = [l for l in all_labels if l != -1]
    else:
        cluster_labels = list(all_labels)

    n_clusters = len(cluster_labels)
    if n_clusters == 0:
        raise ValueError("No clusters to plot (after excluding noise).")

    # color map
    cmap = cm.get_cmap(cmap_name)
    color_cycle = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]

    figs = []
    for i, fid in enumerate(selected_feature_ids):
        feat_name = feature_names[fid] if fid < len(feature_names) else f"feature_{fid}"
        # Global edges for comparability and to avoid the broadcast error
        all_vals = data[np.isin(labels, cluster_labels), fid]
        all_vals = all_vals[np.isfinite(all_vals)]
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
        # layout
        cols = min(max_cols, n_clusters)
        rows = math.ceil(n_clusters / cols)
        fig_w = cols * figsize_per_subplot[0]
        fig_h = rows * figsize_per_subplot[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = axes.flatten()

        for k, lab in enumerate(cluster_labels):
            ax = axes_flat[k]
            idxs = np.where(labels == lab)[0]
            vals = data[idxs, i]
            # filter non-finite values
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "no valid data", ha='center', va='center')
                ax.set_title(f"cluster {lab} (n=0)")
                continue

            color = color_cycle[k % len(color_cycle)]
            #histogram gives error. print some stats
            print(f"cluster {lab}/{n_clusters}, feature {feat_name}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")
            ax.hist(vals, bins=bin_edges, density=True, alpha=0.6, color=color,
                    edgecolor='none', label=f"n={len(idxs)/len(data):.2%}") \
                # median
            med = np.nanmedian(vals)
            ax.axvline(med, color='k', linestyle='--', linewidth=1)
            ax.set_title(f"cluster {lab}")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(False)

        # turn off unused axes
        for j in range(n_clusters, rows*cols):
            axes_flat[j].axis('off')

        fig.suptitle(f"Feature: {feat_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            # sanitize filename
            fname = f"{feat_name}".replace(" ", "_").replace("/", "_")
            outpath = os.path.join(save_dir, f"{fname}.png")
            fig.savefig(outpath, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        figs.append(fig)

    return figs


def plot_transition_matrix(labels, groups, worm_data=None, target_dir="", n_labels=-1, exclude_noise=True, fps=3, allow_self_loops=False):
    labels = np.asarray(labels, dtype=int)
    if exclude_noise:
        clean_indices = np.where(labels >= 0)[0]
        labels = labels[clean_indices]
        groups = groups[clean_indices]

    if n_labels==-1:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

    state_transitions = np.zeros((n_labels, n_labels))

    for g in np.unique(groups):
        group_indices = np.where(groups == g)[0]
        worm_labels = labels[group_indices]
        if worm_data is not None:
            times = worm_data[:, -1][group_indices]  # Assuming worm_data[g] is a tuple where the last element is timestamps
            assert len(times) == len(worm_labels), f"Length of timestamps {len(times)} does not match length of labels {len(worm_labels)} for group {g}"
        for t in range(len(worm_labels) - 1):
            state = worm_labels[t]
            next_state = worm_labels[t + 1]
            if state != next_state or allow_self_loops:
                continuous = True
                if worm_data is not None:
                    time_gap = times[t + 1] - times[t]
                    if time_gap > (2.0 / fps):  # If time gap is larger than expected frame interval, consider it a break
                        continuous = False

                if continuous:
                    state_transitions[state, next_state] += 1

    row_sums = state_transitions.sum(axis=1, keepdims=True)
    counted_transitions = state_transitions.copy()
    np.divide(state_transitions, row_sums, where=row_sums != 0, out=state_transitions)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.grid(False)
    cax = ax.imshow(state_transitions, cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Transition Probability', fontsize=16)
    # increase font size of colorbar
    cbar.ax.tick_params(labelsize=14)

    # Labeling
    # ax.set_title('State Transition Matrix')
    # ax.set_xlabel('Next State')
    # ax.set_ylabel('Current State')
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    '''auto_name_list = ["Slow-line", "Straight-turn", "Loop-turn", "Crawl", "High-turning"]
    auto_name_list = ["Line", "Reversal", "Past pause", "Sharp turn", "4"]
    auto_name_list = ["Orthogonal", "Climb", "Descent", "Reversal", "4"]
    auto_name_list = ["Pause", "Reversal", "Sharp", "Run"]'''

    if "manual" in target_dir:
        ax.set_xticklabels([f"{list(state_dic.keys())[i]}" for i in range(n_labels)])
        ax.set_yticklabels([f"{list(state_dic.keys())[i]}" for i in range(n_labels)])
        '''elif "auto" in target_dir:
        ax.set_xticklabels([auto_name_list[i] for i in unique_labels])
        ax.set_yticklabels([auto_name_list[i] for i in unique_labels])'''
    else:
        ax.set_xticklabels([i for i in range(n_labels)])
        ax.set_yticklabels([i for i in range(n_labels)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    # Annotate with transition probabilities
    for i in range(n_labels):
        for j in range(n_labels):
            value = state_transitions[i, j]
            if value > 0:
                ax.text(j, i, f'{value:.2f}', fontsize=14, ha='center', va='center',
                        color='black')

    plt.tight_layout()
    if len(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, "state_transition_matrix.pdf"))
    else:
        plt.show()

    return state_transitions, counted_transitions


def calculate_angle_changes(angles):

    angle_changes_ = np.diff(angles) # Changes in angle between consecutive frames
    angle_changes_ = [angle_changes_[i] - 2 * np.pi if angle_changes_[i] > np.pi else angle_changes_[i] for i in range(len(angle_changes_))]
    angle_changes_ = [angle_changes_[i] + 2 * np.pi if angle_changes_[i] < -np.pi else angle_changes_[i] for i in range(len(angle_changes_))]

    return angle_changes_

import json

def get_durations(labels):
    durations = {}
    n_labels = len(np.unique(labels))
    if -1 in labels:
        n_labels-=1
    for i in range(n_labels):
        durations[i] = []
    current_duration = 1
    for i in range(len(labels[:-1])):
        state = labels[i]
        if state == -1:
            continue
        next_state = labels[i + 1]
        if state == next_state:
            current_duration += 1
        else:
            if current_duration != 0:
                #check if the key exists, if not, create it
                if int(state) not in durations:
                    durations[int(state)] = []
                durations[int(state)].append(current_duration)
            current_duration = 1
    last_state = labels[-1]
    if last_state != -1:
        if int(last_state) not in durations:
            durations[int(last_state)] = []
        durations[int(last_state)].append(current_duration)

    return durations

def get_durations_continuous_flat(labels, groups, fps=3):
    """Returns a flat array (same len as labels) where each point is assigned its bout duration."""
    durations = np.zeros(len(labels), dtype=int)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        group_labels = labels[idx]

        bout_start = 0
        for i in range(1, len(group_labels)):
            if group_labels[i] != group_labels[bout_start]:
                bout_len = i - bout_start
                durations[idx[bout_start:i]] = bout_len
                bout_start = i

        # Handle last bout
        bout_len = len(group_labels) - bout_start
        durations[idx[bout_start:]] = bout_len

    return durations

chemotaxis_transition_data = json.load(open("../state_estimations/chemotaxis_transitions.json", 'r'))

with open("../auto_agents_100_all_data.json", 'r') as f:
    data = json.load(f)

states = data["sub_states"]
n_worms = data["parameters"]['N']
labels = np.concatenate(states)
#print state frequencies
for i in range(len(np.unique(labels))):
    print(f"state {i}: ", np.sum(labels == i))
    print(f"\t normalised: ", np.sum(labels == i) / len(labels))
    for j in range(len(np.unique(labels))):
        count_ij = np.sum((labels[:-1] == i) & (labels[1:] == j))
        print(f"\t transition {i} -> {j}: ", count_ij)
        print(f"\t normalised: ", count_ij / np.sum(labels == i))
groups = np.repeat(np.arange(n_worms), len(states[0]))


flat_durations = get_durations_continuous_flat(labels, groups)


for g in np.unique(groups):
    worm_durations = flat_durations[groups == g]
    worm_labels = labels[groups == g]
    run_durations = worm_durations[worm_labels == 2]
    xs = list(range(len(run_durations)))
    #plt.plot(xs, run_durations)
    #plt.title(f"off-food worm {g} run durations over time")
    #plt.show()
import sys
#sys.exit()

c = np.array(data["c"])

# --- collect dc_int per (state, next_state)
dc_at_transitions = defaultdict(list)
distances_from_odor_final = []

durations = defaultdict(list)
for g in np.unique(groups):
    g_idx = np.where(groups == g)[0]
    idx = g_idx[g_idx < len(c)]

    worm_c = c[idx]
    worm_labels = labels[g_idx]
    worm_durations = get_durations(worm_labels)
    for state, dur_list in worm_durations.items():
        durations[state].extend(dur_list)
    worm_labels = labels[idx]
    for t in range(len(worm_labels) - 1):
        i = worm_labels[t]
        j = worm_labels[t + 1]

        tau = chemotaxis_transition_data[str(i)][str(j)]["tau"]
        coeff = chemotaxis_transition_data[str(i)][str(j)]["model_coeff"]
        intercept = chemotaxis_transition_data[str(i)][str(j)]["model_intercept"]

        if coeff != -1 and intercept!=-1 and t >= tau:

            if worm_c[t-tau].dtype != np.float64:
                print("warning: missing c value at t-tau, skipping dc_int calculation for this transition")
                print(f"worm {g}, transition {i} -> {j}, t={t}, tau={tau}")
                print("worm_c[t-tau]: ", worm_c[t-tau])
                import sys
                sys.exit()
            dc_int = worm_c[t] - worm_c[t - tau]
            dc_at_transitions[(i, j)].append(dc_int)

#plot durations
for l in np.unique(labels):
    if l==-1:
        continue
    dur_vals = durations.get(l, [])
    dur_vals = np.asarray(dur_vals).flatten()
    if len(dur_vals) == 0:
        continue
    print(f"state {l}, duration: n={len(dur_vals)}, mean={np.mean(dur_vals):.4f}, std={np.std(dur_vals):.4f}, min={np.min(dur_vals):.4f}, max={np.max(dur_vals):.4f}")
    plt.hist(dur_vals, bins=30, density=True, alpha=0.6, color='blue', edgecolor='none')
    plt.xlabel("duration (frames)")
    plt.ylabel("density")
    plt.title(f"State {l} durations")
    plt.grid(False)
    plt.show()



# --- plotting range
dc_min, dc_max = -0.003, 0.003
x = np.linspace(dc_min, dc_max, 200)

N_STATES = len(chemotaxis_transition_data)


speeds = data["velocities"]
speeds = np.array(speeds).flatten()
angles = data["angles"]
angles = np.array(angles).flatten() #WRONG!!! NEED TO TAKE INTO ACCOUNT GROUP SPLIT
angle_changes = calculate_angle_changes(angles)
angle_changes = np.concatenate([[0], angle_changes])  # pad to same length as speeds and labels
print(f"len speeds: {len(speeds)}, len angle_changes: {len(angle_changes)}, len labels: {len(labels)}")
features = np.stack([speeds, angle_changes], axis=1)


plot_feature_distributions_by_cluster(features, labels, feature_names=["speed", "angle_change"])


# --- group by source state
for i in range(N_STATES):

    plt.figure()

    # compute softmax curves for all target states j
    Z = []
    valid_js = []

    for j in range(N_STATES):
        model = chemotaxis_transition_data[str(i)][str(j)]
        tau = model["tau"]
        coeff = model["model_coeff"]
        intercept = model["model_intercept"]

        if coeff == -1 and intercept == -1:
            # fallback: treat as constant prob
            z = model["p_off_food"] * np.ones_like(x)
        else:
            alpha = model["model_coeff"]
            intercept = model["model_intercept"]
            mean = model["mean"]
            std = model["std"]
            sign = model["sign"]
            z = sign*( alpha * (x - mean) / std + intercept)

        Z.append(z)
        valid_js.append(j)

    Z = np.array(Z)  # shape (n_states, len(x))

    # softmax
    expZ = 1/( 1+ np.exp(-Z))
    P = expZ / np.sum(expZ, axis=0, keepdims=True)

    # --- plot curves
    for idx_j, j in enumerate(valid_js):
        plt.plot(x, P[idx_j], label=f"{i}→{j}")

    # --- overlay data (dc_int samples)
    for j in range(N_STATES):
        dc_vals = dc_at_transitions.get((i, j), [])
        dc_vals = np.asarray(dc_vals).flatten()
        if len(dc_vals) == 0:
            continue
        print("transition: ", f"{i}->{j}", np.min(dc_vals), np.max(dc_vals), np.std(dc_vals))

        # jitter vertically around zero for visibility
        y_jitter = np.random.uniform(0, 0.05, size=len(dc_vals))

        plt.scatter(dc_vals, y_jitter, alpha=0.3, s=10)

    plt.xlabel("dc_int")
    plt.ylabel("P(i → j | dc_int)")
    plt.title(f"Source state {i}")
    plt.legend()
    plt.show()



plot_transition_matrix(labels, groups, allow_self_loops=False)