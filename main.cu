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
#include <cstring>

__global__ void initialize_rng(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < WORM_COUNT) {
        // Use a combination of seed, agent ID, and time to ensure unique seeds
        //curand_init(seed + id, 0, 0, &states[id]);
        curand_init(seed, id, 0, &states[id]);
    }
}

void get_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main(int argc, char* argv[]) {
    const char *extracted_params_filename = "/state_estimations/behavior_distributions_off_food.json";
    const char *transition_params_filename = "/state_estimations/l2.json";
    const char *transition_b_params_filename = "/state_estimations/l2b.json";
    const char *exit_params_filename = "/state_estimations/l1.json";
    const char* transition_factors_filename = "/state_estimations/transition_factors.json";
    const char* bias_filename = "/state_estimations/transition_angle_bias.json";
    const char* duration_lognormal_params_filename = "/state_estimations/duration_lognormal_params_all_conditions.json";
    const char* p_roam_filename = "/state_estimations/p_roam_all_conditions.json";
	const char* duration_betaprime_params_filename = "/state_estimations/duration_params.json";
    const char* joint_distribution_file_name = "/state_estimations/joint_distributions_off_food.json";

    //int seed = argc > 1 ? atoi(argv[1]) : SEED;
    const char* output_dir = "/outputs";
    int seed = SEED;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output-dir") == 0 && i+1 < argc)
            output_dir = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            seed = atoi(argv[++i]);
    }

    // create output dir only if length is greater than 0 (i.e. if it was provided as an argument)

    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", output_dir);
    system(mkdir_cmd);

    // build output path
    char output_path[512];
    snprintf(output_path, sizeof(output_path), "%s/auto_agents_100_all_data.json", output_dir);
    // Device-side accumulators (allocated once before the sim loop)
    int*   d_neighbor_sum;    // one int per agent
    int*   d_timestep_count;  // scalar
    cudaMalloc(&d_neighbor_sum, WORM_COUNT * sizeof(int));
    cudaMalloc(&d_timestep_count, sizeof(int));
    cudaMemset(d_neighbor_sum, 0, WORM_COUNT * sizeof(int));
    cudaMemset(d_timestep_count, 0, sizeof(int));
    BehaviorDistributionHost* h_states = new BehaviorDistributionHost[N_STATES];
    TransitionBiasHost* h_biases = new TransitionBiasHost[N_STATES*N_STATES];
    TransitionModelHost* h_transitions = new TransitionModelHost[N_STATES*N_STATES];
    TransitionModelHost* h_transition_b = new TransitionModelHost[N_STATES*N_STATES];
    TransitionModelHost* h_exit = new TransitionModelHost[N_STATES];
    TransitionFactorHost* h_transition_factors = new TransitionFactorHost[N_STATES*N_STATES];
    PRoamHost* h_proam = new PRoamHost;
    Agent* d_agents, *h_agents = new Agent[WORM_COUNT];
    curandState* d_curand_states, *d_states_grids; //?
    size_t size = WORM_COUNT * sizeof(Agent);
    auto* positions = new float[WORM_COUNT * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    auto* angles = new float[WORM_COUNT* N_STEPS]; // Matrix to store angles for each agent at each timestep
    auto* velocities = new float[WORM_COUNT * N_STEPS]; // Matrix to store velocities for each agent at each timestep
    auto* sub_states = new int[WORM_COUNT * N_STEPS]; // Matrix to store substates for each agent at each timestep
	auto* dc = new float[WORM_COUNT * N_STEPS * N_STATES * N_STATES]; // Matrix to store dc_int[tau] for each agent at each timestep for each state transition
	auto* c = new float[WORM_COUNT * N_STEPS]; // Matrix to store chemical concentrations for each agent at each timestep
	float frequencies_host[N_STATES];
    int agent_id = 0;
    char target_json[256];
    char label_sequence_filename[256];
    if (false && argc >= 4) { //first is output dir, second is seed, last, if available, is agent id for single agent logging
        agent_id = atoi(argv[1]);
        snprintf(target_json, sizeof(target_json), "/sim/simulated_worm_%d.json", agent_id);
        snprintf(label_sequence_filename, sizeof(label_sequence_filename),
                 "/state_estimations/off_food_label_sequences/worm_%d_labels.json", agent_id);

        cudaMemcpyToSymbol(d_agent_kappas, agent_kappas, sizeof(agent_kappas));
        cudaMemcpyToSymbol(d_agent_periods, agent_periods, sizeof(agent_periods));
        cudaMemcpyToSymbol(d_agent_amplitudes, agent_amplitudes, sizeof(agent_amplitudes));
      }
    else{
        snprintf(target_json, sizeof(target_json), "/sim/auto_agents_100_all_data.json");
        //snprintf(label_sequence_filename, sizeof(label_sequence_filename),"/state_estimations/off_food_label_sequences/worm_45_labels.json");

     }
	DurationLognormalHost duration_lognormal_params_host[N_STATES], *h_roaming_duration = new DurationLognormalHost;
	StateParams* d_params = nullptr;
	load_distributions(
    duration_betaprime_params_filename,
    joint_distribution_file_name,
    N_STATES,
    &d_params);



    printf("Allocating memory on device...\n");
    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_curand_states, WORM_COUNT * sizeof(curandState));

    initialize_rng<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_curand_states, seed);
    get_last_error();
    cudaDeviceSynchronize();
    initAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, time(NULL), WORM_COUNT, agent_id);
    //printf("Initializing agents\n");
    get_last_error();
    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

    /*load_p_roam(h_proam, p_roam_filename);
    upload_proam(h_proam);
    //load data of single agent into loaded_states
    printf("Loading state data from file...\n");*/
    load_exit_data(h_exit, exit_params_filename);
    upload_exit_models(h_exit);
    load_state_data(h_states, extracted_params_filename, h_proam);
    printf("Uploading state data to device...\n");
    upload_distributions(h_states, h_proam);
    printf("Loading transition data from file...\n");
    load_transition_data(h_transitions, transition_params_filename, frequencies_host);
    printf("Uploading transition data to device...\n");
    upload_transition_models(h_transitions);
    if(TASK=="aggregation-diff"){
        load_transition_data(h_transition_b, transition_b_params_filename, frequencies_host);
        upload_transition_models_b(h_transition_b);
    }
    printf("Loading transition factors from file...\n");
    load_transition_factors(h_transition_factors, transition_factors_filename);
    printf("Uploading transition factors to device...\n");
    upload_transition_factors(h_transition_factors);
    /*printf("Loading bias data from file...\n");
    load_transition_biases(h_biases, bias_filename);
    printf("Uploading bias data to device...\n");
    upload_biases(h_biases);
    load_duration_data(duration_lognormal_params_host, duration_lognormal_params_filename, h_roaming_duration);
    upload_duration_data(duration_lognormal_params_host, h_roaming_duration);*/
    WormLabelSequence seq;
	if(agent_id!=0){
        seq = load_worm_labels_to_device(label_sequence_filename);
    }
    cudaMemcpyToSymbol(odor_x0, &h_odor_x0, sizeof(float));
    cudaMemcpyToSymbol(odor_y0, &h_odor_y0, sizeof(float));
	cudaMemcpyToSymbol(frequencies, frequencies_host, N_STATES * sizeof(float));

    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (int i = 0; i < N_STEPS; ++i) {
        //printf("Step %d\n", i);
        moveAgentsCollective<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, WORM_COUNT, i, d_params);
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

        for (int j = 0; j < WORM_COUNT; ++j) {
            positions[(i * WORM_COUNT + j) * 2] = h_agents[j].x;
            positions[(i * WORM_COUNT + j) * 2 + 1] = h_agents[j].y;

            angles[i * WORM_COUNT + j] = h_agents[j].angle;

            velocities[i * WORM_COUNT + j] = h_agents[j].speed;

            sub_states[i * WORM_COUNT + j] = h_agents[j].state;

            for (int tau = 0; tau < N_STATES; ++tau) {
              for (int tau_next = 0; tau_next < N_STATES; ++tau_next) {
                dc[((i * WORM_COUNT + j) * N_STATES + tau) * N_STATES + tau_next] = h_agents[j].dc_int[tau * N_STATES + tau_next];
              }

            //dc[i * WORM_COUNT + j] = h_agents[j].dc_int[0];
            c[i * WORM_COUNT + j] = h_agents[j].c[0];
        }
        }
        //printf("Updating agent state\n");
        //updateAgentState<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, i, WORM_COUNT, d_params);
        if(agent_id!=0){
            updateAgentStateDeterministic<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_agents, seq.d_labels, seq.length, i);}
        else{
            updateAgentStateCollective<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, i, WORM_COUNT, d_params);
        }
        cudaDeviceSynchronize();

        get_last_error();
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

        accumulate_neighbors<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, WORM_COUNT, d_neighbor_sum, d_timestep_count);


    }
    // After sim loop
    std::vector<int> h_neighbor_sum(WORM_COUNT);
    int h_timestep_count;
    cudaMemcpy(h_neighbor_sum.data(), d_neighbor_sum, WORM_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_timestep_count, d_timestep_count, sizeof(int), cudaMemcpyDeviceToHost);

    float avg_neighbors = 0.0f;
    for (int i = 0; i < WORM_COUNT; i++)
        avg_neighbors += (float)h_neighbor_sum[i] / h_timestep_count;
    avg_neighbors /= WORM_COUNT;
    printf("overall average neighbors per agent: %.2f\n", avg_neighbors);
    /*if(LOG_GENERIC_TARGET_DATA) {
        saveAllDataToJSON(output_path, positions, velocities, angles, h_agents ,WORM_COUNT, N_STEPS, sub_states, dc, c, avg_neighbors);
    }*/
    saveOnlyAvgNeighbors(output_path, avg_neighbors);
    printf("Logging complete to %s\n", output_path);

    printf("Simulation complete. Cleaning up...\n");
    cudaFree(d_agents);
    cudaFree(d_curand_states);
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