//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_AGENT_UPDATE_H
#define UNTITLED_AGENT_UPDATE_H
#include <cuda_runtime.h>
#include <random>
#include <limits>
#include <cmath>

#include "beta_sampling.h"
#include "gaussian_odour.h"
#include "numeric_functions.h"
// Function to sample from a von Mises distribution
__device__ float sample_from_von_mises(float mu, float kappa, curandState* state) {
    // Handle kappa = 0 (uniform distribution)
    if (kappa < 1e-6) {
        return mu + (2.0f * M_PI * curand_uniform(state)) - M_PI; // Random uniform sample
    }

    // Step 1: Setup variables
    float a = 1.0f + sqrt(1.0f + 4.0f * kappa * kappa);
    float b = (a - sqrt(2.0f * a)) / (2.0f * kappa);
    float r = (1.0f + b * b) / (2.0f * b);

    /*std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    */
    while (true) {
        // Step 2: Generate random variables
        float u1 = abs(curand_uniform(state));
        float z = cos(M_PI * u1);
        float f = (1.0f + r * z) / (r + z);
        float c = kappa * (r - f);

        // Step 3: Generate random variable u2
        float u2 = abs(curand_uniform(state));

        // Step 4: Accept/reject step
        if (u2 < c * (2.0f - c) || u2 <= c * exp(1.0f - c)) {
            // Step 5: Generate final angle sample
            float u3 = abs(curand_uniform(state));
            float theta = (u3 < 0.5f) ? acos(f) : -acos(f);
            float result = mu + theta;  // Return the sample from von Mises
            if (result > M_PI) {
                result -= 2.0f * M_PI;
            } else if (result < -M_PI) {
                result += 2.0f * M_PI;
            }
            return result;
        }
    }
}

__device__ int select_next_state(float* probabilities, curandState* local_state, int num_states) {
    // Generate a random value between 0 and 1
    float random_val = curand_uniform(local_state);

    // Cumulative probability tracking
    float cumulative_prob = 0.0f;

    // Iterate through probabilities to select state
    for (int i = 0; i < num_states; i++) {
        cumulative_prob += probabilities[i];
        // If random value is less than cumulative probability, select this state
        if (random_val <= cumulative_prob) {
            return i;
        }
    }
    //printf("Error: No state selected\n");
    // Fallback to random state if no state is selected (should rarely happen)
    return (int) curand_uniform(&local_state[0]) * N_STATES;;
}



__global__ void moveAgents(Agent* agents, curandState* local_state, State* states, int worm_count) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {
        int agent_state = agents[agent_id].state;
        curandState local_rng = local_state[agent_id];
        State state = states[agent_id * N_STATES + agent_state];
        /*
        //log x,y index before moving for count grid
        int cur_x_index = static_cast<int>(agents[agent_id].x / DX);
        int cur_y_index = static_cast<int>(agents[agent_id].y / DY);

        //printf("\tsampling angle...\n");

        float sampled_angle= sample_from_von_mises(state.angle_mu, state.angle_kappa, &local_state[agent_id]);
        if (repulsive_pheromone[cur_x_index*N+cur_y_index]>POTENTIAL_THRESHOLD) {
            printf("current phero: %f, sampling\n", repulsive_pheromone[cur_x_index*N+cur_y_index]);
            float min_repulsive_phero_direction = sampled_angle;
            float min_repulsive_phero = repulsive_pheromone[cur_x_index*N+cur_y_index];
            for (int i = 0; i < 32; ++i) {
                float angle = curand_uniform(&local_state[agent_id]) * 2 * M_PI;
                int sample_x = static_cast<int> (cur_x_index + DX * cosf(angle));
                int sample_y = static_cast<int> (cur_y_index + DY * sinf(angle));
                if (repulsive_pheromone[sample_x*N+sample_y]<=min_repulsive_phero) {
                    min_repulsive_phero = repulsive_pheromone[sample_x*N+sample_y];
                    min_repulsive_phero_direction = angle;
                }
            }
            sampled_angle = 1e2f * min_repulsive_phero * min_repulsive_phero_direction + (1.0f - 1e2f * min_repulsive_phero) * min_repulsive_phero_direction;
        }*/

        //pick randomly between 0 and pi based on the angle_change_mix: 0 = 0, 1 = pi

        float random_uniform_sample = curand_uniform(&local_state[agent_id]), sampled_angle;
        float target_kappa = state.angle_kappa;
        if (abs(state.angle_change_mix - 0.5f)<0.2f) { //in almost equal proportions, the fitting will have a very wide kappa
            target_kappa = 10.0f;
        }
        if (random_uniform_sample>state.angle_change_mix) {
            float target_angle = M_PI;
            /*if (state.angle_mu > 1.0f) {
                target_angle = state.angle_mu;
            }*/
            //printf("random sample %f > angle change mix %f\n", random_uniform_sample, state.angle_change_mix);
            sampled_angle = state.angle_mu_sign *  sample_from_von_mises(target_angle, target_kappa, &local_state[agent_id]);

            while (std::abs(sampled_angle)<1.0f) {
                printf("Resampling angle change %f\n", sampled_angle);
                sampled_angle = state.angle_mu_sign *  sample_from_von_mises(target_angle, target_kappa, &local_state[agent_id]);
                printf("New angle change: %f\n", sampled_angle);
            }

            if (agent_state == 1) {
                if (std::abs(sampled_angle)<1.0f) {
                    printf("sampled a suspicious angle %f\n", sampled_angle);
                    printf("agent %d in state %d\n", agent_id, agent_state);
                    printf("state %d has angle_mu = %f, angle_kappa = %f, angle_change_mix = %f\n", agent_state, state.angle_mu, state.angle_kappa, state.angle_change_mix);
                }
            }

        }
        else {
            float target_angle = 0.0f;

            //printf("random sample %f < angle change mix %f\n", random_uniform_sample, state.angle_change_mix);
            sampled_angle =state.angle_mu_sign *  sample_from_von_mises(target_angle,target_kappa, &local_state[agent_id]);


        }
        //float sampled_angle= sample_from_von_mises(state.angle_mu_sign * state.angle_mu, state.angle_kappa, &local_state[agent_id]);

        //printf("sampled angle = %f\n", sampled_angle);
        agents[agent_id].angle+=sampled_angle;
        if(agents[agent_id].angle>2 * M_PI || agents[agent_id].angle<-2 * M_PI){
            agents[agent_id].angle = fmodf(agents[agent_id].angle, 2*M_PI);
        }
        //printf("\tsampling speed...\n");

        float sampled_speed =( sample_beta_device(&local_state[agent_id], state.speed_alpha, state.speed_beta))* (state.max_speed - state.min_speed) + fmaxf(0.0f, state.min_speed);

        //float sampled_speed =( sample_beta_device(&local_state[agent_id], state.speed_alpha, state.speed_beta) + fmaxf(state.min_speed, state.speed_loc))* state.speed_scale;
        while (sampled_speed>state.max_speed || sampled_speed<state.min_speed) {
            sampled_speed = (sample_beta_device(&local_state[agent_id], state.speed_alpha, state.speed_beta) ) * (state.max_speed - state.min_speed)+ fmaxf(0.0f, state.min_speed);
            //printf("with state %d resampled speed : %f\n", agent_state, sampled_speed);
        }

        //apply DT for speed
        sampled_speed *= DT;

        float fx = cosf(agents[agent_id].angle);
        float fy = sinf(agents[agent_id].angle);

        float dx = fx * sampled_speed;
        float dy = fy * sampled_speed;


        //float prev_x = agents[agent_id].x , prev_y = agents[agent_id].y;

        //move
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;

        //apply periodic boundary conditions
        if (agents[agent_id].x < 0) agents[agent_id].x += WIDTH;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x -= WIDTH;
        if (agents[agent_id].y < 0) agents[agent_id].y += HEIGHT;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y -= HEIGHT;

        //check if the new x or y indices are different, if so, -1 to the previous grid point, +1 to the new one
        /*int next_x_index = static_cast<int>(agents[agent_id].x / DX);
        int next_y_index = static_cast<int>(agents[agent_id].y / DY);
        if (next_x_index!=cur_x_index || next_y_index!=cur_y_index) {
            //printf("agent changed cell: from (%f,%f) in (%d,%d) to (%f, %f) in (%d, %d) with DX=DY=%f\n", prev_x, prev_y, cur_x_index, cur_y_index, agents[agent_id].x, agents[agent_id].y, next_x_index, next_y_index, DX);
            atomicAdd(&agent_count_grid[cur_x_index*N+cur_y_index], -1);
            atomicAdd(&agent_count_grid[next_x_index*N+next_y_index], 1);
        }*/

        agents[agent_id].speed = sampled_speed;
        //printf("speed : %f\n", sampled_speed);




    }
}

__global__ void updateAgentState(Agent* agents, curandState* local_state, State* states, int timestep, int worm_count) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {
        curandState local_rng = local_state[agent_id];
        int agent_state = agents[agent_id].state;
        //printf("agent state %d\n", agent_state);

        State state = states[agent_id * N_STATES + agent_state];

        int x_index =static_cast<int>(agents[agent_id].x/DX), y_index = static_cast<int>(agents[agent_id].y/DY);

        //float potential_bacteria = bacterial_lawn[x_index*N+y_index];
//BACTERIAL_ATTRACTION_STRENGTH * log(bacterial_lawn[x_index*N+y_index]+BACTERIAL_ATTRACTION_SCALE);
        //float repulsive_potential = - repulsive_pheromone[x_index*N+y_index];
//REPULSIVE_PHEROMONE_STRENGTH * log(repulsive_pheromone[x_index*N+y_index]+REPULSIVE_PHEROMONE_SCALE);
        //float cur_potential = potential_bacteria + repulsive_potential;
        //float delta_potential = cur_potential - agents[agent_id].previous_potential;

        //agents[agent_id].cumulative_potential += cur_potential;
        //printf("bacterial density=%f (p=%f) repulsive phero=%f (p=%f) cum = %f\n", bacterial_lawn[x_index*N+y_index], potential_bacteria, repulsive_pheromone[x_index*N+y_index], repulsive_potential, agents[agent_id].cumulative_potential);
        if (agents[agent_id].state_duration>0){// && abs(delta_potential)<=POTENTIAL_THRESHOLD) {
            agents[agent_id].state_duration--;
            //printf("diminishing duration\n");
        }
        else{
            if (agents[agent_id].state_duration!=-1) { //not the initialisation value, need to transition
                float probabilities[N_STATES];
                float sum_of_probabilities=0.0f;
                for (int i = 0; i < N_STATES; i++) {
                    if (i==agent_state)
                        probabilities[i] = 0.0f;
                    else {
                        probabilities[i] = fmaxf((states[agent_id * N_STATES + i].probability_m * static_cast<float>(timestep ) / (120.0f/ DT) + states[agent_id * N_STATES + i].probability_q), 0.0f);// / (120.0f / DT), 0.0f);
                        probabilities[i]*=  fmaxf(states[agent_id * N_STATES + agent_state].transition_likelihood[i], 0.0f);

                    }
                    sum_of_probabilities+=probabilities[i];
                }
                if (sum_of_probabilities>0.0f) {
                    for (int i = 0; i < N_STATES; i++) {
                        probabilities[i]/=sum_of_probabilities;
                    }
                }
                else {
                    printf("Error in probability computation! Good luck.\n");
                    printf("\tSome debug info:\n");
                    printf("\tagent state: %d\n", agent_state);
                    printf("\ttimestep: %d", timestep);
                    printf("\tprobabilities: ");
                    for (int i = 0; i < N_STATES; i++) {
                        printf("\t state probabilities[%d]=%f\n", i, probabilities[i]);
                    }
                    printf("\t m=%f q=%f\n", state.probability_m, state.probability_q);
                    //since the probabilities are all 0, just set them to uniform, except the current probb
                    //printf("not in the good case, but should be fine?\n");
                    for (int i = 0; i < N_STATES; i++) {
                        if (i!=agent_state) {
                            probabilities[i] = 1.0f/(N_STATES-1.0f);
                        }
                        //printf("constant probability to state %d is %f\n", i, probabilities[i]);

                    }
                }
                int next_state = select_next_state(probabilities, &local_state[agent_id], N_STATES);
                    //printf("new state selected %d\n", next_state);
                agents[agent_id].state = next_state;
                state = states[agent_id * N_STATES + next_state];
            }
            int duration;
            if (state.duration_alpha==-1.0f and state.duration_beta==-1.0f) {
                duration = 0;
            }
            else {
                float scale = state.duration_scale;

                /*if (state.breaking_point>0 && timestep % state.breaking_point==0) {
                    if ( (timestep / state.breaking_point) % 2 ==1){
                        scale = state.duration_scale1;}
                    else{
                        scale = state.duration_scale2;}
                }*/
                float s = sample_beta_device(&local_state[agent_id], state.duration_alpha, state.duration_beta);
                //duration = static_cast<int>(round((s + fmaxf(0.0f, state.duration_loc - 1)) * scale));
                duration = static_cast<int>(round(s * scale + state.duration_loc));

                //printf("sampled new duration (float) %f + loc %f * scale %f = %d\n", s, state.duration_loc, scale, duration);

                while (duration<0 || duration>=N_STEPS) {
                    float nvidia_beta_sample = sample_beta_device(&local_state[agent_id], state.duration_alpha, state.duration_beta);
                    duration =  static_cast<int>(round((nvidia_beta_sample  + fmaxf(0.0f, state.duration_loc))* scale));
                    //printf("sampled new duration (float) %f + loc %f * scale %f = %d\n", nvidia_beta_sample, state.duration_loc, scale, duration);

                }

            }
            agents[agent_id].state_duration=duration;

            if(curand_uniform(&local_state[agent_id])>0.5){
                state.angle_mu_sign = -1;
            }else {
                state.angle_mu_sign = 1;
            }
            //printf("new sign: %d\n", state.angle_mu_sign);


        }
        //printf("agent state: %d with duration: %d\n", agents[agent_id].state, agents[agent_id].state_duration);
        //agents[agent_id].previous_potential = cur_potential;
    }
}




// CUDA kernel to update the position of each agent
__global__ void moveAgents2(Agent* agents, curandState* states, ExplorationState * d_explorationStates,  /*float* potential, int* agent_count_grid,*/ int worm_count, int timestep, float sigma) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        float max_concentration_x = 0.0f;
        float max_concentration_y = 0.0f;
        float sensed_potential = computeDensityAtPoint(agents[id].x, agents[id].y, timestep);//potential[agent_x * N + agent_y];
        sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
        //add a small perceptual noise to the potential
        if(sigma!=0.0f){
            float perceptual_noise = curand_normal(&states[id]) * sigma;
            if(perceptual_noise>sigma) perceptual_noise = sigma;
            if(perceptual_noise<-sigma) perceptual_noise = (-sigma);
            sensed_potential += perceptual_noise;
        }

        float max_concentration = sensed_potential;
        //printf("Sensed potential: %f\n", sensed_potential);
        for (int i = 0; i < 32; ++i) {
            float angle = curand_uniform(&states[id]) * 2 * M_PI;
            float sample_x = agents[id].x + SENSING_RADIUS * cosf(angle);
            float sample_y = agents[id].y + SENSING_RADIUS * sinf(angle);
            float concentration = computeDensityAtPoint(sample_x, sample_y, timestep);
            // Add perceptual noise if sigma is not zero
            if (sigma != 0.0f) {
                concentration += curand_normal(&states[id]) * sigma;
            }
            concentration = ATTRACTION_STRENGTH * logf(concentration + ATTRACTION_SCALE);

            if (concentration > max_concentration) {
                max_concentration = concentration;
                max_concentration_x = cosf(angle);
                max_concentration_y = sinf(angle);
            }
        }

        float auto_transition_probability = curand_uniform(&states[id]);
        if (agents[id].cumulative_potential > PIROUETTE_TO_RUN_THRESHOLD || auto_transition_probability>=AUTO_TRANSITION_PROBABILITY_THRESHOLD){ //|| auto_transition_probability>=AUTO_TRANSITION_PROBABILITY_THRESHOLD){ //starting to move in the "right" direction, then RUN
            agents[id].state = 0;
            agents[id].cumulative_potential = 0.0f;
        }
        else if (sensed_potential - agents[id].previous_potential < -ODOR_THRESHOLD){ //|| auto_transition_probability<AUTO_TRANSITION_PROBABILITY_THRESHOLD){ //moving in the wrong direction, then PIROUETTE
            agents[id].state = 1;
            agents[id].cumulative_potential += (sensed_potential - agents[id].previous_potential);
        }

        float fx, fy, new_angle;
        float mu, kappa, scale, shape;
        int sub_state = agents[id].substate;
        int base_index = id * N_STATES;
        ExplorationState* explorationState = &d_explorationStates[base_index + sub_state];
        float* probabilities = explorationState->probabilities;
        mu = explorationState->angle_mu;    // this can be negative, set below, 50% chance
        kappa = explorationState->angle_kappa;
        scale = explorationState->speed_scale;
        shape = explorationState->speed_spread;
        float random_angle = (float)explorationState->angle_mu_sign * sample_from_von_mises(mu, kappa, states);//wrapped_cauchy(0.0, 0.6, &states[id]);//curand_normal(&states[id]) * M_PI/4;////
        //float random_angle = sample_from_von_mises(mu, kappa, states);//wrapped_cauchy(0.0, 0.6, &states[id]);//curand_normal(&states[id]) * M_PI/4;////

        //printf("sampled angle: %f with mu=%f kappa=%f sign=%d\n", random_angle, mu, kappa, explorationState->angle_mu_sign);
        float lambda=0.0f; //@TODO: try to make it a function of the potential (and re-add the potential)

        if(agents[id].state == 0){ //if the agent is moving = RUN - LOW TURNING - EXPLOIT
            //if the max concentration is 0 (or best direction is 0,0), then choose random only (atan will give unreliable results)
            if (max_concentration< ODOR_THRESHOLD || (max_concentration_x==0 && max_concentration_y==0) ) {

                agents[id].angle += ((1.0f-lambda)* random_angle);
            }
            else {
                float norm = sqrt(max_concentration_x * max_concentration_x + max_concentration_y * max_concentration_y);
                float direction_x = max_concentration_x / norm;
                float direction_y = max_concentration_y / norm;
                float bias = atan2(direction_y, direction_x);

                float current_angle = agents[id].angle;
                if(bias-current_angle>=0){
                    bias = M_PI / 4;
                } else{
                    bias = -M_PI / 4;
                }

                float k = KAPPA;// * pow(sensed_potential / max_concentration, 2);
                new_angle = sample_from_von_mises(bias, k, &states[id]);

                agents[id].angle += new_angle;
            }

        }
        else{ //BROWNIAN MOTION - HIGH TURNING - EXPLORE
            agents[id].angle += random_angle;

        }

        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }

        fx = cosf(agents[id].angle);
        fy = sinf(agents[id].angle);

        float new_speed = curand_log_normal(&states[id], logf(scale), shape);
        while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //printf("New Speed: %f with scale %f and shape %f\n", new_speed, scale, shape);

        float dx = fx * new_speed;
        float dy = fy * new_speed;

        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].speed = new_speed;
        // Apply periodic boundary conditions
        if (agents[id].x < 0) agents[id].x += WIDTH;
        if (agents[id].x >= WIDTH) agents[id].x -= WIDTH;
        if (agents[id].y < 0) agents[id].y += HEIGHT;
        if (agents[id].y >= HEIGHT) agents[id].y -= HEIGHT;
        int new_x = (int)(agents[id].x / DX);
        int new_y = (int)(agents[id].y / DY);


        //printf("Current substate: %d, max duration %d id again (just to be sure) %d\n", agents[id].substate, explorationState->max_duration, explorationState->id);

        agents[id].substate = select_next_state(probabilities, &states[id], N_STATES);
            //printf("switching to state %d\n", agents[id].substate);
        explorationState = &d_explorationStates[base_index + agents[id].substate];

        //IF the new substate is different from the previous one, then choose sign for the angle mu and augments timesteps in this substate, otherwise set to 0
        if(agents[id].substate != agents[id].previous_substate){
            if(curand_uniform(&states[id])>0.5){
                explorationState->angle_mu_sign *= -1;
            }
            explorationState->timesteps_in_state++;
        } else {
            explorationState->timesteps_in_state = 0;
        }

        agents[id].previous_substate = sub_state;
        //check if the agent is in the target area
        if (new_x >= 3* N/4 - TARGET_AREA_SIDE_LENGTH/2 && new_x < 3*N/4 + TARGET_AREA_SIDE_LENGTH/2 && new_y >= N/2 - TARGET_AREA_SIDE_LENGTH/2 && new_y < N/2 + TARGET_AREA_SIDE_LENGTH/2){
            agents[id].is_agent_in_target_area = 1;
            agents[id].steps_in_target_area++;
            if(agents[id].first_timestep_in_target_area == -1){
                agents[id].first_timestep_in_target_area = timestep;
            }
        }
        else{
            agents[id].is_agent_in_target_area = 0;
        }

    }
}
#endif //UNTITLED_AGENT_UPDATE_H
