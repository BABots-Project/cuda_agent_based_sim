#!/bin/bash

set -e

# === Configuration ===
IMAGE_NAME=cuda-agent-sim
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHOD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

METHODS=("KMeans" "HDBSCAN" "Ward" "Spectral")
if [[ -n "$METHOD" ]]; then
    METHODS=("$METHOD")
fi

STATE_ESTIMATIONS_HOST="${SCRIPT_DIR}/state_estimations"
STATE_ESTIMATIONS_CONTAINER=/state_estimations
SIMULATION_CONTAINER=/sim
PERMS_BASE="${STATE_ESTIMATIONS_HOST}/permutations"

if [[ -z "$(docker images -q cuda-env:12.2 2> /dev/null)" ]]; then
    echo "Loading base image..."
    docker load -i cuda-env.tar
else
    echo "Base image already present."
fi

echo "Building Docker image: $IMAGE_NAME"
DOCKER_BUILDKIT=1 docker build --pull=false -t cuda-agent-sim .
docker image prune -f
TOTAL=$(ls -d "${PERMS_BASE}"/perm_* | wc -l)
CURRENT=0
# iterate over all perm_* folders in sorted order
for PERM_DIR in $(ls -d "${PERMS_BASE}"/perm_* | sort); do
    CURRENT=$((CURRENT + 1))
    echo "=== Permutation ${CURRENT}/${TOTAL} ==="
    PERM=$(basename "$PERM_DIR")
    PERM_ID="${PERM#perm_}"   # strip "perm_" prefix → "000", "001", ...

    SPLIT_FILE="${PERM_DIR}/split.json"
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "No split.json in ${PERM_DIR}, skipping"
        continue
    fi

    TEST_IDS=$(python3 -c "
import json
with open('${SPLIT_FILE}') as f:
    d = json.load(f)
print(' '.join(str(x) for x in d['test']))
")
    echo "=== Permutation ${PERM_ID} | test worms: ${TEST_IDS} ==="

    for METHOD in "${METHODS[@]}"; do
        echo "--- Method: $METHOD ---"

        PERM_SRC="${PERM_DIR}/${METHOD}"
        if [[ ! -d "$PERM_SRC" ]]; then
            echo "Missing ${PERM_SRC}, skipping"
            continue
        fi

        JOINT_PATH="${STATE_ESTIMATIONS_CONTAINER}/off_food_${METHOD}_train_joint_distributions.json"

        cp "${PERM_SRC}/train_joint_distributions.json" \
           "${STATE_ESTIMATIONS_HOST}/off_food_${METHOD}_train_joint_distributions.json"
        cp "${PERM_SRC}/transition_matrix.json" \
           "${STATE_ESTIMATIONS_HOST}/transition_matrix_perm_${PERM_ID}.json"
        cp "${PERM_SRC}/swing_params.json" \
           "${STATE_ESTIMATIONS_HOST}/swing_params_perm_${PERM_ID}.json"

        OUTPUT_HOST="${SCRIPT_DIR}/sim_results/${PERM}/${METHOD}"
        mkdir -p "$OUTPUT_HOST"

        for i in $TEST_IDS; do
            echo "  Running agent $i..."
            docker run --rm --privileged --gpus all \
                -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
                -v "$OUTPUT_HOST":"$SIMULATION_CONTAINER" \
                "$IMAGE_NAME" --agent "$i" --joint "$JOINT_PATH" --method "$METHOD"
        done

        echo "  Done: ${METHOD} → ${OUTPUT_HOST}"
    done

    echo "=== Permutation ${PERM_ID} complete ==="
done

echo "All permutations done. Results in ${SCRIPT_DIR}/sim_results/"