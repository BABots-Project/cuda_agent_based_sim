#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ESTIMATIONS_HOST="${SCRIPT_DIR}/state_estimations"
SIMULATION_HOST="${SCRIPT_DIR}"
STATE_ESTIMATIONS_CONTAINER=/state_estimations
SIMULATION_CONTAINER=/sim
IMAGE_NAME=cuda-agent-sim

if [[ -z "$(docker images -q cuda-env:12.2 2>/dev/null)" ]]; then
    echo "Loading base image..."
    docker load -i cuda-env.tar
fi

docker build --pull=false -t "$IMAGE_NAME" "$SCRIPT_DIR"

docker run --rm --privileged --gpus all \
    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
    "$IMAGE_NAME" "$@"   # forward all args to the binary