#include <stdio.h>
#include <curand_kernel.h>
#include "include/json.hpp"
#include <fstream>
#include <iostream>
#include "headers/parameters.h"
#include "headers/init_env.h"
#include "headers/agent_update.h"
#include "headers/update_matrices.h"
#include "headers/logging.h"
#include "headers/gaussian_odour.h"

__global__ void initialize_rng(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < WORM_COUNT) {
        // Use a combination of seed, agent ID, and time to ensure unique seeds
        curand_init(seed + id, 0, 0, &states[id]);
    }
    //curand_init(seed + id, id, 0, &states[id]); // Unique seed for each thread
}

void get_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main(int argc, char* argv[]) {
    //const char *extracted_params_filename = "/home/nema/PycharmProjects/behavioral_flagging2/state_estimations/auto_data.json";
    //const char* target_json = "/home/nema/cuda_agent_based_sim/auto_agents_100_all_data.json";
    const char *extracted_params_filename = "/state_estimations/auto_data.json";
    const char* target_json = "/sim/auto_agents_1000_all_data.json";

    State* d_states, *h_states = new State[N_STATES * WORM_COUNT];
    State* loaded_states = new State[N_STATES];
    Agent* d_agents, *h_agents = new Agent[WORM_COUNT];
    curandState* d_curand_states, *d_states_grids;
    size_t size = WORM_COUNT * sizeof(Agent);
    auto* positions = new float[WORM_COUNT * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    auto* angles = new float[WORM_COUNT* N_STEPS]; // Matrix to store angles for each agent at each timestep
    auto* velocities = new float[WORM_COUNT * N_STEPS]; // Matrix to store velocities for each agent at each timestep
    auto* sub_states = new int[WORM_COUNT * N_STEPS]; // Matrix to store substates for each agent at each timestep

    //float* d_repulsive_pheromone, * h_repulsive_pheromone = new float[N*N]; //device and host grids for the repulsive pheromone secreted
    //float* d_bacterial_lawn, *h_bacterial_lawn = new float[N*N]; //device and host grids for the bacterial density (food)
    //int* d_agent_count, *h_agent_count = new int[N*N]; //device and host grids for the number of agents inside each grid segment

    //float * repulsive_pheromone_history = new float[N*N*N_STEPS];
    //float * bacterial_lawn_history = new float[N*N*N_STEPS];

    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_curand_states, WORM_COUNT * sizeof(curandState));
    cudaMalloc(&d_states, N_STATES * sizeof(State) * WORM_COUNT);
    /*cudaMalloc(&d_repulsive_pheromone, N*N*sizeof(float));
    cudaMalloc(&d_bacterial_lawn, N*N*sizeof(float));
    cudaMalloc(&d_agent_count, N*N*sizeof(int));*/

    initialize_rng<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_curand_states, SEED);
    get_last_error();
    cudaDeviceSynchronize();
    initAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, time(NULL), WORM_COUNT);
    //printf("Initializing agents\n");
    get_last_error();
    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

    //load data of single agent into loaded_states
    load_state_data(loaded_states, extracted_params_filename);
    //h_states needs to be set by sampling from loaded_states
    populate_plausible_states(loaded_states, h_states);
    cudaMemcpy(d_states, h_states, WORM_COUNT*N_STATES*sizeof(State), cudaMemcpyHostToDevice);

    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    //initialise the agent count grid
    /*initAgentDensityGrid<<<gridSize, blockSize>>>(d_agent_count, d_agents, WORM_COUNT);
    get_last_error();
    cudaDeviceSynchronize();
    cudaMemcpy(h_agent_count, d_agent_count, N*N*sizeof(int), cudaMemcpyDeviceToHost);*/

    //initialise the repulsive pheromone grid
    /*initRepulsivePheromoneGrid<<<gridSize, blockSize>>>(d_repulsive_pheromone, d_agent_count);
    get_last_error();
    cudaDeviceSynchronize();
    cudaMemcpy(h_repulsive_pheromone, d_repulsive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);*/

    //initialise the bacterial lawn
    /*float x_circle_center = WIDTH/2, y_circle_center = HEIGHT/2;
    float delta_angle = 4 * M_PI / N_FOOD_SPOTS;
    int x_centers[N_FOOD_SPOTS], y_centers[N_FOOD_SPOTS];
    for (int i=0; i < N_FOOD_SPOTS; i++) {
        float x = x_circle_center + SPOT_DISTANCE * cos(delta_angle * static_cast<float>(i));
        float y = y_circle_center + SPOT_DISTANCE * sin(delta_angle * static_cast<float>(i));
        int x_index = static_cast<int>(x/WIDTH * N);
        int y_index = static_cast<int>(y/HEIGHT * N);
        x_centers[i] = x_index;
        y_centers[i] = y_index;
        printf("x_index, y_index %d = %d, %d\n", i, x_index, y_index);

    }
    initBacterialLawnSquareHost(h_bacterial_lawn, x_centers, y_centers);
    cudaMemcpy(d_bacterial_lawn, h_bacterial_lawn, N*N*sizeof(float), cudaMemcpyHostToDevice);*/

    for (int i = 0; i < N_STEPS; ++i) {
        printf("Step %d\n", i);
        moveAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, d_states, WORM_COUNT);
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_agent_count, d_agent_count, N*N*sizeof(int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < WORM_COUNT; ++j) {
            positions[(i * WORM_COUNT + j) * 2] = h_agents[j].x;
            positions[(i * WORM_COUNT + j) * 2 + 1] = h_agents[j].y;

            angles[i * WORM_COUNT + j] = h_agents[j].angle;

            velocities[i * WORM_COUNT + j] = h_agents[j].speed;

            sub_states[i * WORM_COUNT + j] = h_agents[j].state;
        }
        //printf("Updating state...\n");
        updateAgentState<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, d_states, i, WORM_COUNT);
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

        //update the grids
        /*cudaMemcpy(d_repulsive_pheromone, h_repulsive_pheromone, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bacterial_lawn, h_bacterial_lawn, N*N*sizeof(float), cudaMemcpyHostToDevice);
        updateRepulsivePheromoneAndBacterialLawnGrids<<<gridSize, blockSize>>>(d_repulsive_pheromone, d_bacterial_lawn, d_agent_count);
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_repulsive_pheromone, d_repulsive_pheromone, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bacterial_lawn, d_bacterial_lawn, N*N*sizeof(float), cudaMemcpyDeviceToHost);

        //log the grids
        if (LOG_REPULSIVE_PHEROMONE) {
            logMatrixToFile("/home/nema/cuda_agent_based_sim/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, N, N, i);
        }
        if (LOG_BACTERIAL_LAWN) {
            logMatrixToFile("/home/nema/cuda_agent_based_sim/logs/bacterial_lawn/bacterial_lawn_step_", h_bacterial_lawn, N, N, i);

        }
        logIntMatrixToFile("/home/nema/cuda_agent_based_sim/logs/agent_count/agent_count_step_", h_agent_count, N, N, i);
*/

    }
    if(LOG_GENERIC_TARGET_DATA) {
        saveAllDataToJSON(target_json, positions, velocities, angles, h_agents ,WORM_COUNT, N_STEPS, sub_states);
    }

    cudaFree(d_agents);
    cudaFree(d_curand_states);
    cudaFree(d_states);
    /*cudaFree(d_repulsive_pheromone);
    cudaFree(d_bacterial_lawn);
    cudaFree(d_agent_count);*/
    delete[] positions;
    delete[] angles;
    delete[] velocities;
    delete[] sub_states;
    delete[] h_agents;
    delete[] h_states;
    /*delete[] h_repulsive_pheromone;
    delete[] h_bacterial_lawn;
    delete[] h_agent_count;
    delete[] repulsive_pheromone_history;
    delete[] bacterial_lawn_history;*/
}