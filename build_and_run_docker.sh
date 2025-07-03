#!/bin/bash

set -e

# === Configuration ===
IMAGE_NAME=cuda-agent-sim
DOCKERFILE_PATH=.

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define paths relative to the script's directory
STATE_ESTIMATIONS_HOST="${SCRIPT_DIR}/state_estimations"
SIMULATION_HOST="${SCRIPT_DIR}"

STATE_ESTIMATIONS_CONTAINER=/state_estimations
SIMULATION_CONTAINER=/sim

# === Build Docker Image ===
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$DOCKERFILE_PATH"

# === Run the container ===
echo "Running container with GPU support..."
docker run --rm --privileged --gpus all \
    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
    "$IMAGE_NAME"

