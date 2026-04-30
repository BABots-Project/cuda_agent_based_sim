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

__device__ int select_next_state(
    float* probabilities,
    curandState* rng,
    int num_states)
{
    float r = curand_uniform(rng);

    float cumulative = 0.0f;

    for (int i = 0; i < num_states; i++)
    {
        cumulative += probabilities[i];

        if (r <= cumulative)
            return i;
    }

    // fallback if rounding errors occur
    return num_states - 1;
}

__device__ float sample_betaprime(float alpha, float beta, float scale,
                                   curandState* rng) {
    // X ~ Gamma(alpha), Y ~ Gamma(beta), then X/Y ~ BetaPrime(alpha,beta)
    float x = sample_gamma_device(rng, alpha);
    float y = sample_gamma_device(rng, beta);
    return (x / y) * scale;
}

// ---- Alias draw from a JointTable ------------------------------------
__device__ void alias_draw(const JointTable* table, curandState* rng,
                            float* out_speed, float* out_angle) {
    int   i = (int)(curand_uniform(rng) * table->n);   // uniform bin
    float u =       curand_uniform(rng);
    int   idx = (u < table->prob[i]) ? i : table->alias[i];
    *out_speed = table->obs[idx * 2];
    *out_angle = table->obs[idx * 2 + 1];
}

// ---- Interpolated draw (point iv) ------------------------------------
__device__ void draw_speed_angle(const StateParams* sp, int t_star,
                                  curandState* rng,
                                  float* out_speed, float* out_angle) {
    // Binary search for t0, t1 bracketing t_star
    int lo = 0, hi = sp->n_durations - 1;

    // exact match
    // (linear scan is fine for small n_durations; replace with bsearch if needed)
    for (int i = 0; i < sp->n_durations; i++) {
        if (sp->durations[i] == t_star) {
            alias_draw(&sp->tables[i], rng, out_speed, out_angle);
            return;
        }
    }

    // find bracketing t0, t1
    int idx0 = 0;
    while (idx0 < sp->n_durations - 1 && sp->durations[idx0 + 1] < t_star)
        idx0++;
    int idx1 = idx0 + 1;

    // clamp to edges (extrapolation → nearest)
    if (t_star < sp->durations[0]) {
        alias_draw(&sp->tables[0], rng, out_speed, out_angle);
        return;
    }
    if (t_star > sp->durations[sp->n_durations - 1]) {
        alias_draw(&sp->tables[sp->n_durations - 1], rng, out_speed, out_angle);
        return;
    }

    float t0 = sp->durations[idx0];
    float t1 = sp->durations[idx1];
    float lambda = (t_star - t0) / (t1 - t0);   // weight toward t1

    // stochastic interpolation: draw from t0 or t1 with prob (1-l, l)
    if (curand_uniform(rng) > lambda)
        alias_draw(&sp->tables[idx0], rng, out_speed, out_angle);
    else
        alias_draw(&sp->tables[idx1], rng, out_speed, out_angle);
}

__device__ float sample_von_mises(curandState* rng, float kappa) {
    // Best & Fisher (1979) algorithm
    // Returns a sample in (-pi, pi) with concentration kappa around 0

    float tau  = 1.0f + sqrtf(1.0f + 4.0f * kappa * kappa);
    float rho  = (tau - sqrtf(2.0f * tau)) / (2.0f * kappa);
    float r    = (1.0f + rho * rho) / (2.0f * rho);

    float z, f, c, u1, u2, u3;
    while (true) {
        u1 = curand_uniform(rng);
        u2 = curand_uniform(rng);
        u3 = curand_uniform(rng);

        z  = cosf(3.14159265f * u1);
        f  = (1.0f + r * z) / (r + z);
        c  = kappa * (r - f);

        if (c * (2.0f - c) > u2) break;           // acceptance condition 1
        if (logf(c / u2) + 1.0f - c >= 0.0f) break;  // acceptance condition 2
    }

    return (u3 > 0.5f ? 1.0f : -1.0f) * acosf(f);
}


__global__ void moveAgentsCollective(Agent* agents, curandState* local_state, int worm_count, int timestep, StateParams* params){
  int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {
      int agent_state = agents[agent_id].state;
		int neighbor_count = 0;
        for (int j = 0; j < WORM_COUNT; j++) {
            if (agent_id == j) continue;
            float a_dx = agents[agent_id].x - agents[j].x;
            float a_dy = agents[agent_id].y - agents[j].y;
            float dist = sqrtf(a_dx*a_dx + a_dy*a_dy);
            if (dist < 0.5f) neighbor_count++;
        }

        // you're already computing this — just store it
        agents[agent_id].neighbor_count = neighbor_count;

        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
    	draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
		float mu_score = 0.536f, std_score = 0.547f;
        float mu_period = 2.363f, sigma_period = 0.581f;
        if (agent_state == 2) {
            // initialize once when entering run
    		if (agents[agent_id].previous_state != 2 || timestep==0) {
                  if(agents[agent_id].agent_id ==0){
        				//float zP = curand_normal(&local_rng);
        				//float sampled_period = roundf(expf(mu_period + sigma_period * zP));
        				//if (sampled_period < 4)  sampled_period = 4;
        				//if (sampled_period > 60) sampled_period = 60;

        				agents[agent_id].run_omega = 2.0f * 3.14159265f / 8.0f;//sampled_period; //

        				/*float zA = curand_normal(&local_rng);
        				float a = mu_score + std_score * zA;//
        				if (a < 0.213f) a = 0.213f;
        				if (a > 0.850f) a = 0.850f;*/
        				agents[agent_id].run_amp = 0.55f;
					}
        		agents[agent_id].phi = 0.0f;//sample_von_mises(&local_rng, 1.5f);// sample_von_mises(&local_rng, agents[agent_id].kappa);//2.0f * 3.14159265f * curand_uniform(&local_rng);
    		}

    // phase noise makes the oscillation less rigid
    		float sigma_phi = 0.0f;//0.6934f;   // tune from data
    		agents[agent_id].phi += agents[agent_id].run_omega + sigma_phi * curand_normal(&local_rng);

    // mean-zero angle noise widens the distribution around 0
    		float sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa); // tune from residuals of real data
    		angle_change = agents[agent_id].run_amp * sinf(agents[agent_id].phi)
                  + sigma_theta;
            //do not exceed +/-1.5rad
            //scale from [-pi, pi] to [-1.5, 1.5]
            angle_change /= 2.0f;
		}

        float new_angle =agents[agent_id].angle+angle_change;

        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
		if (new_angle < 0) new_angle += 2 * M_PI;
		new_angle -= M_PI;

        //density dependent linear speed    modulation
        float alpha_speed = 0.1f; //
        float speed_factor = fmaxf(0.5f, 1.0f - alpha_speed * (float)agents[agent_id].neighbor_count); //clip to avoid negative speed
        speed *= speed_factor;

		//clip speed to 0-MAXIMUM_ALLOWED_SPEED
        if(speed<0.0f) speed=0.0f;
        if(speed>MAX_ALLOWED_SPEED) speed=MAX_ALLOWED_SPEED;


        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;

        agents[agent_id].x += dx;
        agents[agent_id].y += dy;

        //apply periodic boundary conditions
        if (agents[agent_id].x < 0) agents[agent_id].x += WIDTH;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x -= WIDTH;
        if (agents[agent_id].y < 0) agents[agent_id].y += HEIGHT;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y -= HEIGHT;

        agents[agent_id].previous_speed = agents[agent_id].speed;
        agents[agent_id].previous_angle = agents[agent_id].angle_change;
        agents[agent_id].previous_mag_angle_change = fabsf(agents[agent_id].angle_change);

        agents[agent_id].speed = speed;
        agents[agent_id].angle = new_angle;
        agents[agent_id].angle_change = angle_change;



		local_state[agent_id] = local_rng;

    }
}

__global__ void updateAgentStateCollective(
    Agent* agents,
    curandState* rng_states,
    int timestep,
    int worm_count, StateParams* params)
{
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (agent_id >= worm_count)
        return;

    if(agents[agent_id].state_duration>1 && agents[agent_id].state==2 && agents[agent_id].neighbor_count>0){ //only consider early exit for run state
      TransitionModel exit_model = d_exit_models[agents[agent_id].state];
      //use exit model to determine if the agent should exit the state early -- it's a logistic function on the number of neighbors
        float p_exit = exit_model.height / (1.0f + expf(-exit_model.coeff * (float)agents[agent_id].neighbor_count + exit_model.intercept));
        float u = curand_uniform(&rng_states[agent_id]);
        if (u < p_exit) {
          //set duration to 0
            agents[agent_id].state_duration = 0;
        }
    }

    if(agents[agent_id].state_duration > 1){

      agents[agent_id].previous_state = agents[agent_id].state;
      agents[agent_id].state = agents[agent_id].state; //keep the same state
      agents[agent_id].state_duration -= 1;
        return; //don't update state if duration not over
    }

    curandState local_rng = rng_states[agent_id];

    int agent_state = agents[agent_id].state;

    float p[N_STATES];

	float p_irr = 0.0f;
	float p_r_raw[N_STATES];
	float sum_r = 0.0f;

	// PASS 1: compute raw values
	for (int i = 0; i < N_STATES; i++)
	{
    	const TransitionModel& model =
        	d_transition_models[agent_state * N_STATES + i];

    	if ((model.coeff==-1 && model.intercept==-1) || agents[agent_id].neighbor_count<1)
    	{
          	//printf("No neighbors or no model for transition %d->%d (agent %d, neighbors=%d)\n", agent_state, i, agent_id, agents[agent_id].neighbor_count);
        	p[i] = model.p_off_food;
        	p_irr += p[i];
        	p_r_raw[i] = 0.0f; // important
    	}
    	else
    	{

        	float z = model.coeff *  (float) agents[agent_id].neighbor_count + model.intercept;
        	float height = model.height;  // new field in TransitionModel
            float val = height / (1.0f + expf(-z));
        	p_r_raw[i] = val;
        	sum_r += val;
    	}
	}

	// PASS 2: normalize ONLY relevant transitions
	float remaining_mass = 1.0f - p_irr;

	if (sum_r > 0.0f && remaining_mass > 0.0f)
	{
    	for (int i = 0; i < N_STATES; i++)
    	{
        	const TransitionModel& model =
            	d_transition_models[agent_state * N_STATES + i];

        	if (!(model.coeff==-1 && model.intercept==-1 ) || agents[agent_id].neighbor_count>0)//|| fabsf(agents[agent_id].accumulated_dc_tot) < ODOR_THRESHOLD))
        	{
            	p[i] = (p_r_raw[i] / sum_r) * remaining_mass;
        	}
    	}
	}


    int next_state = select_next_state(p, &local_rng, N_STATES);
    if (next_state < 0 || next_state >= N_STATES) {
    	printf("ERROR next_state=%d (agent %d)\n", next_state, agent_id);
    	return;
	}

	agents[agent_id].previous_state = agents[agent_id].state;
    agents[agent_id].state = next_state;
    //sample duration for the new state
    const BehaviorDistribution& new_state_dist = d_behavior_distributions[next_state];
    float u = curand_uniform(&local_rng);
	int idx = 0;
	for (int j = 0; j < new_state_dist.n_duration_bins - 1; j++) {
    	if (u <= new_state_dist.duration_prob[j]) {
        	idx = j;
        	break;
    	}
    	idx = j + 1;  // fallback to last bin if u > all but last cumprob
	}
	int new_duration = new_state_dist.duration_bins[idx];
	agents[agent_id].state_duration = max(new_duration, 1); //at least 1 timestep in the new state

      BehaviorDistribution state = d_behavior_distributions[next_state];
      agents[agent_id].p_same_sign = state.p_same_sign;


	agents[agent_id].initial_state_duration = agents[agent_id].state_duration;

    rng_states[agent_id] = local_rng;
}


__global__ void moveAgents(Agent* agents, curandState* local_state, int worm_count, int timestep, StateParams* params) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {
        /*

        BehaviorDistribution state = d_behavior_distributions[agent_state];
		int n_speed_bins;
         float *speed_bins, *speed_prob;
         int *speed_alias;
        if(agents[agent_id].is_persistent && STATE_MAX_DURATIONS[agents[agent_id].state]>d_proam.thresh){
          		n_speed_bins=state.n_roaming_speed_bins;
                speed_bins = state.roaming_speed_bins;
                speed_prob = state.roaming_speed_prob;
                speed_alias = state.roaming_speed_alias;
          }
        else{
           n_speed_bins=state.n_speed_bins;
                speed_bins = state.speed_bins;
                speed_prob = state.speed_prob;
                speed_alias = state.speed_alias;
        }
        //sample speed
        int k = (int)(curand_uniform(&local_rng) * n_speed_bins);

        if (k >= n_speed_bins)
            k = n_speed_bins - 1;
        float r = curand_uniform(&local_rng);

        int idx = (r < speed_prob[k]) ? k : speed_alias[k];
        float speed_raw = speed_bins[idx];

        //float speed = state.speed_alpha * agents[agent_id].previous_speed + (1.0f-fabsf(state.speed_alpha)) * speed_raw;
		float speed = speed_raw;
        //ar(1)
        //float speed = state.speed_alpha * (agents[agent_id].previous_speed - state.speed_mean) + speed_raw + state.speed_mean;
        //in case a transition occurred, use only sampled speed


        //float speed = speed_raw;
         //sample angle change

         float lambda;
         int n_angle_bins;
         float *angle_bins, *angle_prob;
         int *angle_alias;
            if(agents[agent_id].is_persistent && STATE_MAX_DURATIONS[agents[agent_id].state]>d_proam.thresh){
                lambda = state.angle_alpha;
                n_angle_bins = state.roaming_n_angle_bins;
                angle_bins = state.roaming_angle_bins;
                angle_prob = state.roaming_angle_prob;
                angle_alias = state.roaming_angle_alias;
            }
            else {
                lambda = 0.0f;
                n_angle_bins = state.n_angle_bins;
                angle_bins = state.angle_bins;
                angle_prob = state.angle_prob;
                angle_alias = state.angle_alias;
            }


        k = curand_uniform(&local_rng) * n_angle_bins;
        if (k >= n_angle_bins)
            k = n_angle_bins - 1;
        r = curand_uniform(&local_rng);
        idx = (r < angle_prob[k]) ? k : angle_alias[k];
        float angle_change_raw = angle_bins[idx];


        mag *= ac_factor;

        //angle_change_raw *= ac_factor;


        float angle_change;// =lambda * agents[agent_id].previous_angle + (1.0f-lambda) * angle_change_raw;
		//ar(1)
        //angle_change = state.angle_alpha * (agents[agent_id].previous_angle - state.angle_mean) + angle_change + state.angle_mean;
		if(agents[agent_id].previous_state != agent_state) {
            angle_change = angle_change_raw;
            //printf("Agent %d: state transition from %d to %d,  angle change raw %f, angle change %f\n", agent_id, agents[agent_id].previous_state, agent_state, angle_change_raw, angle_change);
        }
        else {
			lambda = curand_normal(&local_rng) * state.std_angular_difference + state.mean_angular_difference;
            mag = lambda + agents[agent_id].previous_angle;
            //pick sign based on state.p_same_sign
            float sign=1.0f;
            if(curand_uniform(&local_rng) < state.p_same_sign){
                sign = copysignf(1.0f, agents[agent_id].previous_angle);
            }
            else {
                sign = copysignf(1.0f, -agents[agent_id].previous_angle);
            }
            angle_change = mag;
            //printf("Agent %d: no state transition, angle change raw %f, angle change %f, lambda %f, previous mag angle change %f\n", agent_id, angle_change_raw, angle_change, lambda, agents[agent_id].previous_mag_angle_change);
        }
		angle_change =  angle_change_raw;// * (- 2.0f/(3.0f *(float)STATE_MAX_DURATIONS[agent_state]) * (float)(agents[agent_id].initial_state_duration-1) + 1.0f);
		if(agents[agent_id].state==1 && agents[agent_id].is_persistent && agent_state==agents[agent_id].previous_state) angle_change = copysignf(angle_change, -agents[agent_id].previous_angle);
        ;// lambda*agents[agent_id].angle+(1-lambda)*angle_change;

        */
        //if dc_int is > 0: limit the angle change, by scaling it
        //for now, compute dc_int simply as the instantaneous dc = c[0] - c[1]
                //find tau for prev state -> current state transition
                int agent_state = agents[agent_id].state;

        const TransitionModel& model = d_transition_models[agents[agent_id].previous_state * N_STATES + agent_state];
        float dc;
        if(model.intercept != -1 && model.coeff != -1){
            dc = agents[agent_id].c[0] - agents[agent_id].c[model.tau];
        }
        else {
            dc = agents[agent_id].c[0] - agents[agent_id].c[1];
        }

        const TransitionBias& bias = d_transition_biases[agents[agent_id].previous_state * N_STATES + agent_state];
        const TransitionFactor& factor = d_transition_factors[agents[agent_id].previous_state * N_STATES + agent_state];
        float ac_factor = 1.0f, sp_factor = 1.0f;
        float mag = 1.0f;//fabsf(angle_change_raw);
        if(dc>0.0f){
            ac_factor = factor.angle_plus;
            sp_factor = factor.speed_plus;

            //angle_change += ac_factor;
        } else if (dc<0.0f){
            ac_factor = factor.angle_minus;
            sp_factor = factor.speed_minus;
            //angle_change += ac_factor;
        }
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
    	draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
		float mu_score = 0.536f, std_score = 0.547f;
        float mu_period = 2.363f, sigma_period = 0.581f;
        if (agent_state == 2) {
    // initialize once when entering run
    		if (agents[agent_id].previous_state != 2 || timestep==0) {
                  if(agents[agent_id].agent_id ==0){
        				float zP = curand_normal(&local_rng);
        				float sampled_period = roundf(expf(mu_period + sigma_period * zP));
        				if (sampled_period < 4)  sampled_period = 4;
        				if (sampled_period > 60) sampled_period = 60;

        				agents[agent_id].run_omega = 2.0f * 3.14159265f / sampled_period; //

        				float zA = curand_normal(&local_rng);
        				float a = mu_score + std_score * zA;//
        				if (a < 0.213f) a = 0.213f;
        				if (a > 0.850f) a = 0.850f;
        				agents[agent_id].run_amp = a;
					}
        		agents[agent_id].phi = 0.0f;//sample_von_mises(&local_rng, 1.5f);// sample_von_mises(&local_rng, agents[agent_id].kappa);//2.0f * 3.14159265f * curand_uniform(&local_rng);
    		}

    // phase noise makes the oscillation less rigid
    		float sigma_phi = 0.0f;//0.6934f;   // tune from data
    		agents[agent_id].phi += agents[agent_id].run_omega + sigma_phi * curand_normal(&local_rng);

    // mean-zero angle noise widens the distribution around 0
    		float sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa); // tune from residuals of real data
    		angle_change = agents[agent_id].run_amp * sinf(agents[agent_id].phi)
                  + sigma_theta;
		}

        //apply sign-based dc factor
        angle_change *= ac_factor;
        speed *= sp_factor;

        float new_angle =agents[agent_id].angle+angle_change;
        //if (agent_state == 0) //reversal state 0
    	//new_angle += M_PI;
        //keep between -pi and pi
        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
		if (new_angle < 0) new_angle += 2 * M_PI;
		new_angle -= M_PI;
		//clip speed to 0-MAXIMUM_ALLOWED_SPEED
        if(speed<0.0f) speed=0.0f;
        if(speed>MAX_ALLOWED_SPEED) speed=MAX_ALLOWED_SPEED;


        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;

        //move -- only if the agent is NOT within a 1mm radius of the odor source, as if it was glued
        float odor_dist = sqrtf((agents[agent_id].x - odor_x0) * (agents[agent_id].x - odor_x0) + (agents[agent_id].y - odor_y0) * (agents[agent_id].y - odor_y0));
        if(odor_dist > 1.0f){
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;
        //printf("Agent %d is NOT within 1mm of the odor source, moving. distance = %f, dx=%f, dy=%f, speed=%f\n", agent_id, odor_dist, dx, dy, speed);
        }
        //else{
          //printf("Agent %d is within 1mm of the odor source, not moving. distance = %f\n", agent_id, odor_dist);
          //}

        //apply periodic boundary conditions
        /*if (agents[agent_id].x < 0) agents[agent_id].x += WIDTH;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x -= WIDTH;
        if (agents[agent_id].y < 0) agents[agent_id].y += HEIGHT;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y -= HEIGHT;*/
        //just keep them within the boundaries for now
        if (agents[agent_id].x < 0) agents[agent_id].x = 0;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x = WIDTH - 0.001f;
        if (agents[agent_id].y < 0) agents[agent_id].y = 0;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y = HEIGHT - 0.001f;

        agents[agent_id].previous_speed = agents[agent_id].speed;
        agents[agent_id].previous_angle = agents[agent_id].angle_change;
        agents[agent_id].previous_mag_angle_change = fabsf(agents[agent_id].angle_change);

        agents[agent_id].speed = speed;
        agents[agent_id].angle = new_angle;
        agents[agent_id].angle_change = angle_change;

        //update sensing history: first, compute the sensed dC value at the current position
        //then, shift the history and add the new value at the end
        float sensed_concentration = diffusionProfile(agents[agent_id].x, agents[agent_id].y, timestep * DT);
        //first value is the most recent, last value is the oldest
        for (int i = 99; i > 0; i--) {
            agents[agent_id].c[i] = agents[agent_id].c[i-1];
        }
        agents[agent_id].c[0] = sensed_concentration;

        if (sensed_concentration != 0.0f) {
            printf("Agent %d sensed concentration %f at position (%f, %f) at time %f, dc %f\n", agent_id, sensed_concentration, agents[agent_id].x, agents[agent_id].y, timestep * DT, dc);
        }
        //update accumulated dc total
        //compute dcs for each lag
        float accumulated_dc_tot = 0.0f;
        for (int tau1=0; tau1<N_STATES; tau1++){
            for (int tau2=0; tau2<N_STATES; tau2++){
                int idx = tau1 * N_STATES + tau2;
                const TransitionModel& model = d_transition_models[idx];
                if(model.tau != -1){
                    agents[agent_id].dc_int[idx] = agents[agent_id].c[model.tau] - agents[agent_id].c[0];
                    accumulated_dc_tot += agents[agent_id].dc_int[idx];
                }
                else {
                    agents[agent_id].dc_int[idx] = 0.0f;
                }
            }
        }

        agents[agent_id].accumulated_dc_tot = accumulated_dc_tot;

        int neighbor_count = 0;
        for (int j = 0; j < WORM_COUNT; j++) {
            if (agent_id == j) continue;
            float a_dx = agents[agent_id].x - agents[j].x;
            float a_dy = agents[agent_id].y - agents[j].y;
            float dist = sqrtf(a_dx*a_dx + a_dy*a_dy);
            if (dist < 1.0f) neighbor_count++;  // 1mm radius
        }

        // you're already computing this — just store it
        agents[agent_id].neighbor_count = neighbor_count;

		local_state[agent_id] = local_rng;

    }
}

__global__ void accumulate_neighbors(Agent* agents, int n_agents,
                                     int* neighbor_sum, int* timestep_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    atomicAdd(&neighbor_sum[i], agents[i].neighbor_count);

    // only one thread increments the timestep counter
    if (i == 0) atomicAdd(timestep_count, 1);
}

__global__ void updateAgentState(
    Agent* agents,
    curandState* rng_states,
    int timestep,
    int worm_count, StateParams* params)
{
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (agent_id >= worm_count)
        return;

    //printf("test\n");
    //printf("state duration %d\n", agents[agent_id].state_duration);
    if(agents[agent_id].state_duration > 1){
      agents[agent_id].previous_state = agents[agent_id].state;
      agents[agent_id].state = agents[agent_id].state; //keep the same state
      agents[agent_id].state_duration -= 1;
        return; //don't update state if duration not over
    }

    curandState local_rng = rng_states[agent_id];

    int agent_state = agents[agent_id].state;

    float p[N_STATES];

	float p_irr = 0.0f;
	float p_r_raw[N_STATES];
	float sum_r = 0.0f;

	// PASS 1: compute raw values
	for (int i = 0; i < N_STATES; i++)
	{
    	const TransitionModel& model =
        	d_transition_models[agent_state * N_STATES + i];

        	/*if (model.tau < 0 || model.tau >= 1000) {
    			printf("ERROR tau=%d invalid\n", model.tau);
    			return;
			}*/

    	if ((model.coeff==-1 && model.intercept==-1 )|| fabsf(agents[agent_id].accumulated_dc_tot) < ODOR_THRESHOLD || model.tau==-1)
    	{
        	p[i] = model.p_off_food;
        	p_irr += p[i];
        	p_r_raw[i] = 0.0f; // important
    	}
    	else
    	{
        	float dc_int_value = agents[agent_id].c[0] - agents[agent_id].c[model.tau];
			dc_int_value = (dc_int_value - model.mean) / model.std; //z-score normalization
        	float z = model.sign * (model.coeff * dc_int_value + model.intercept); //add a bit of gain
        	//float val = 1.0f / (1.0f + expf(-z));
			float val = expf(z);
        	p_r_raw[i] = val;
        	sum_r += val;
    	}
	}

	// PASS 2: normalize ONLY relevant transitions
	float remaining_mass = 1.0f - p_irr;

	if (sum_r > 0.0f && remaining_mass > 0.0f)
	{
    	for (int i = 0; i < N_STATES; i++)
    	{
        	const TransitionModel& model =
            	d_transition_models[agent_state * N_STATES + i];

        	if (!(model.coeff==-1 && model.intercept==-1 ))//|| fabsf(agents[agent_id].accumulated_dc_tot) < ODOR_THRESHOLD))
        	{
            	p[i] = (p_r_raw[i] / sum_r) * remaining_mass;
        	}
    	}
	}


    int next_state = select_next_state(p, &local_rng, N_STATES);
    if (next_state < 0 || next_state >= N_STATES) {
    	printf("ERROR next_state=%d (agent %d)\n", next_state, agent_id);
    	return;
	}

	if (STATE_MAX_DURATIONS[next_state] <= 0) {
    	printf("ERROR invalid max duration for state %d\n", next_state);
    	return;
	}

	agents[agent_id].previous_state = agents[agent_id].state;
    agents[agent_id].state = next_state;
    //sample duration for the new state
    const BehaviorDistribution& new_state_dist = d_behavior_distributions[next_state];
    float u = curand_uniform(&local_rng);
	int idx = 0;
	for (int j = 0; j < new_state_dist.n_duration_bins - 1; j++) {
    	if (u <= new_state_dist.duration_prob[j]) {
        	idx = j;
        	break;
    	}
    	idx = j + 1;  // fallback to last bin if u > all but last cumprob
	}
	int new_duration = new_state_dist.duration_bins[idx];
	/*DurationLognormal duration_params = d_duration_lognormals[next_state];
    float mu = duration_params.mu, sigma = duration_params.sigma;
    int loc=0;
    agents[agent_id].is_persistent = false;
    if(duration_params.sigma == -1.0f && duration_params.mu == -1.0f){
        new_duration = 1; //if no duration data, just stay 1 timestep in the new state
    }else{
      if(STATE_MAX_DURATIONS[next_state] > d_proam.thresh){
          //have to flip a coin
          float u = curand_uniform(&local_rng);
          if (u<d_proam.p_roam){
            //sample from the roam distribution
            agents[agent_id].is_persistent = true;
            mu = d_roaming_lognormal.mu;
            sigma = d_roaming_lognormal.sigma;
            loc = d_proam.thresh;
            }
        }

    do {
    new_duration =(int) roundf( + expf(mu + sigma * curand_normal(&local_rng)));
	} while (new_duration > STATE_MAX_DURATIONS[next_state] - 1);

	StateParams* sp = &params[next_state];
       do{
    new_duration = (int)roundf(sample_betaprime(sp->bp_alpha, sp->bp_beta, sp->bp_scale, &local_rng));
    } while (new_duration > STATE_MAX_DURATIONS[next_state]-1);
	*/
	agents[agent_id].state_duration = max(new_duration, 1); //at least 1 timestep in the new state

      BehaviorDistribution state = d_behavior_distributions[next_state];
      agents[agent_id].p_same_sign = state.p_same_sign;


	agents[agent_id].initial_state_duration = agents[agent_id].state_duration;

    rng_states[agent_id] = local_rng;
}


__global__ void updateAgentStateDeterministic(
        Agent* agents,
        const int* __restrict__ d_labels,
        int   n_labels,
        int   t)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= WORM_COUNT) return;

    // clamp t so we freeze at the last label if the simulation runs longer
    int t_cur  = min(t,     n_labels - 1);
    int t_prev = max(0, t - 1);

    int current_state  = d_labels[t_cur];
    int previous_state = d_labels[t_prev];

    agents[id].previous_state = agents[id].state;
    agents[id].state          = current_state;

    if (previous_state != current_state) {
        // ── state just changed: count the run length ahead ────────────────
        int run_length = 0;
        for (int tau = t_cur; tau < n_labels; tau++) {
            if (d_labels[tau] == current_state) run_length++;
            else                                break;
        }
        agents[id].initial_state_duration = run_length;
        agents[id].state_duration         = run_length;
    } else {
        // ── continuing in same state ──────────────────────────────────────
        agents[id].state_duration -= 1;
    }
}


#endif //UNTITLED_AGENT_UPDATE_H
