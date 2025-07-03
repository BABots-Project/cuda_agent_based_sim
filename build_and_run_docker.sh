#!/bin/bash

set -e

# === Configuration ===
IMAGE_NAME=cuda-agent-sim
DOCKERFILE_PATH=.
STATE_ESTIMATIONS_HOST=/home/nema/cuda_agent_based_sim/state_estimations
SIMULATION_HOST=/home/nema/cuda_agent_based_sim
STATE_ESTIMATIONS_CONTAINER=/state_estimations
SIMULATION_CONTAINER=/sim

# === Build Docker Image ===
echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME $DOCKERFILE_PATH

# === Run the container ===
echo "Running container with GPU support..."
docker run --rm --privileged --gpus all \
    -v "$STATE_ESTIMATIONS_HOST":"$STATE_ESTIMATIONS_CONTAINER" \
    -v "$SIMULATION_HOST":"$SIMULATION_CONTAINER" \
    $IMAGE_NAME
