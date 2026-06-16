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
    		//do not exceed [-1.5, 1.5]
            /*while(fabsf(sigma_theta)>1.5f){
                sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa);
               }*/
			sigma_theta /= 2.0f;
                angle_change = agents[agent_id].run_amp * sinf(agents[agent_id].phi)
                  + sigma_theta;
            //do not exceed +/-1.5rad

		}

        float new_angle =agents[agent_id].angle+angle_change;

        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
		if (new_angle < 0) new_angle += 2 * M_PI;
		new_angle -= M_PI;

        //density dependent linear speed    modulation
        float alpha_speed = 0.1f; //
        float speed_factor = fmaxf(0.1f, 1.0f - alpha_speed * (float)agents[agent_id].occlusion_neighbor_count); //clip to avoid negative speed
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

		int neighbor_count = 0, occlusion_count = 0;
        for (int j = 0; j < WORM_COUNT; j++) {
            if (agent_id == j) continue;
            float a_dx = agents[agent_id].x - agents[j].x;
            float a_dy = agents[agent_id].y - agents[j].y;
            float dist = sqrtf(a_dx*a_dx + a_dy*a_dy);
            if (dist < SENSING_RADIUS) neighbor_count++;
            if (dist < OCCLUSION_RADIUS) occlusion_count++;
        }

        // you're already computing this — just store itv
        agents[agent_id].delta_neighbor_count = agents[agent_id].neighbor_count - agents[agent_id].prev_neighbor_count;
        agents[agent_id].prev_neighbor_count  = agents[agent_id].neighbor_count;
        agents[agent_id].neighbor_count = neighbor_count;
        agents[agent_id].occlusion_neighbor_count = occlusion_count;
        //if(timestep==0) printf("Agent %d has %d neighbors\n", agent_id, neighbor_count);
        //if(timestep==1799) printf("Agent %d has %d neighbors at the end\n", agent_id, neighbor_count);

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

    if(agents[agent_id].state_duration>1 && agents[agent_id].state==2 ){//&& agents[agent_id].neighbor_count>0){ //only consider early exit for run state
      TransitionModel exit_model = d_exit_models[agents[agent_id].state];
      //use exit model to determine if the agent should exit the state early -- it's a logistic function on the number of neighbors
        float p_exit = exit_model.height / (1.0f + expf(-exit_model.coeff * (float)agents[agent_id].delta_neighbor_count + exit_model.intercept));
        //float p_exit = exit_model.height / (1.0f + expf(-exit_model.coeff * (float)agents[agent_id].neighbor_count + exit_model.intercept));

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

    	if ((model.coeff==-1 && model.intercept==-1))// || agents[agent_id].neighbor_count<1)
    	{
          	//printf("No neighbors or no model for transition %d->%d (agent %d, neighbors=%d)\n", agent_state, i, agent_id, agents[agent_id].neighbor_count);
        	p[i] = model.p_off_food;
        	p_irr += p[i];
        	p_r_raw[i] = 0.0f; // important
    	}
    	else
    	{

        	float z = model.coeff *  (float) agents[agent_id].delta_neighbor_count + model.intercept;
        	//float z = model.coeff *  (float) agents[agent_id].neighbor_count + model.intercept;

            float height = model.height;  // new field in TransitionModel
            float val = height / (1.0f + expf(-z));
            if(TASK=="aggregation-diff"){
                 const TransitionModel& model_b = d_transition_models_b[agent_state * N_STATES + i];
                    float z_b = model_b.coeff *  (float) agents[agent_id].neighbor_count + model_b.intercept;
                    float height_b = model_b.height;
                    float val_b = height_b / (1.0f + expf(-z_b));
                    val -= val_b;
            }
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

        	if (!(model.coeff==-1 && model.intercept==-1 ))// || agents[agent_id].neighbor_count>0)//|| fabsf(agents[agent_id].accumulated_dc_tot) < ODOR_THRESHOLD))
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
        int agent_state = agents[agent_id].state;
        if(timestep==0){
        	//compute initial concentration
        	agents[agent_id].c[0] = diffusionProfile(agents[agent_id].x, agents[agent_id].y, timestep * DT);
            agents[agent_id].c[1] =0.0f;
            //printf("p1 minus: %f\n", chemotaxis_params_d.p1_minus);
        }
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
    	draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
		float mu_score = 0.536f, std_score = 0.547f, min_score = 0.213f, max_score = 0.850f;
        float mu_score_chemotaxis = 0.61f, std_score_chemotaxis = 0.13f, min_score_chemotaxis = 0.3f, max_score_chemotaxis=0.9f;
        float mu_period = 2.363f, sigma_period = 0.581f;
        float mu_period_chemotaxis = 2.21f, sigma_period_chemotaxis = 0.67f, min_period_chemotaxis = 6.0f, max_period_chemotaxis = 70.0f;
        if (agent_state == 2) {
            // initialize once when entering run
    		if (agents[agent_id].previous_state != 2 || timestep==0) {
                  if(agents[agent_id].agent_id ==0){
        				float zP = curand_normal(&local_rng);
        				float sampled_period = roundf(expf(mu_period_chemotaxis + sigma_period_chemotaxis * zP));
        				if (sampled_period < min_period_chemotaxis)  sampled_period = min_period_chemotaxis;
        				if (sampled_period > max_period_chemotaxis) sampled_period = max_period_chemotaxis;
						//float periods[2] = {10.0f, 28.0f};
                       	//float amplitudes[2] = {0.4611468230085233f, 0.2889931168760453f};
                        //float u = curand_uniform(&local_rng);
                        sampled_period = 8.0f; //median of chemotaxis data
        				agents[agent_id].run_omega = 2.0f * 3.14159265f / sampled_period; //

        				float zA = curand_normal(&local_rng);
        				float a = mu_score_chemotaxis + std_score_chemotaxis * zA;//
        				if (a < min_score_chemotaxis) a = min_score_chemotaxis;
        				if (a > max_score_chemotaxis) a = max_score_chemotaxis;
        				agents[agent_id].run_amp = 0.63f; //median of chemotaxis data
					}
        		agents[agent_id].phi = 0.0f;//sample_von_mises(&local_rng, 1.5f);// sample_von_mises(&local_rng, agents[agent_id].kappa);//2.0f * 3.14159265f * curand_uniform(&local_rng);
    		}

    // phase noise makes the oscillation less rigid
    		float sigma_phi = 0.0f;//0.6934f;   // tune from data
    		agents[agent_id].phi += agents[agent_id].run_omega;// + sigma_phi * curand_normal(&local_rng);

    // mean-zero angle noise widens the distribution around 0
    		float sigma_theta = sample_von_mises(&local_rng, 3.0f);// agents[agent_id].kappa); // tune from residuals of real data
    		//do not exceed [-1.5, 1.5]
            /*while(fabsf(sigma_theta)>1.0f){
                sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa);
               }*/
			sigma_theta /= 3.0f;
                angle_change = agents[agent_id].run_amp * sinf(agents[agent_id].phi)
                  + sigma_theta;

		}
		float ac_factor=1.0f, speed_factor=1.0f;
        float dc = agents[agent_id].c[0] - agents[agent_id].c[1];
        /*if(agent_state == 0 && dc>0) {speed_factor=0.39909376023558557f; ac_factor=3.2603546869951985f;}
        if(agent_state == 0 && dc<0) {speed_factor*=1.0392585990928724f; ac_factor=0.7711346057467615f;}
		if(agent_state == 2 && dc>0) {speed_factor*=0.9931069017531974f; ac_factor*=1.0245742585308728f;}
        if(agent_state == 2 && dc<0) {speed_factor*=1.0656365159518908f; ac_factor*=0.8504060249624477f;}
		speed*=speed_factor;
        angle_change*=ac_factor;*/
        float new_angle =agents[agent_id].angle+angle_change;

        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
		if (new_angle < 0) new_angle += 2 * M_PI;
		new_angle -= M_PI;
		//clip speed to 0-MAXIMUM_ALLOWED_SPEED
        if(speed<0.0f) speed=0.0f;
        if(speed>MAX_ALLOWED_SPEED) speed=MAX_ALLOWED_SPEED;


        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;

        //update position
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;

        //apply periodic boundary conditions
        /*if (agents[agent_id].x < 0) agents[agent_id].x += WIDTH;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x -= WIDTH;
        if (agents[agent_id].y < 0) agents[agent_id].y += HEIGHT;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y -= HEIGHT;*/
        //just keep them within the boundaries for now
        if (agents[agent_id].x < 0){
            //printf("T=%d agent %d has x<0: %f, setting to 0\n", timestep, agent_id,agents[agent_id].x);
            agents[agent_id].x = 0;
        }
        if (agents[agent_id].x >= WIDTH){
            //printf("T=%d agent %d has x>WIDTH: %f, setting to WIDTH-0.001f\n", timestep, agent_id,agents[agent_id].x);
            agents[agent_id].x = WIDTH - 0.001f;
        }
        if (agents[agent_id].y < 0){
            //printf("T=%d agent %d has y<0: %f, setting to 0\n", timestep, agent_id,agents[agent_id].y);
            agents[agent_id].y = 0;
        }
        if (agents[agent_id].y >= HEIGHT){
            //printf("T=%d agent %d has y>HEIGHT: %f, setting to HEIGHT-0.001f\n", timestep, agent_id,agents[agent_id].y);
            agents[agent_id].y = HEIGHT - 0.001f;
        }

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
        //but we just care about value 0 (now) and 1 (previous)
        //so old 0 becomes 1, and sensed value becomes new 0
        agents[agent_id].c[1] = agents[agent_id].c[0];
        agents[agent_id].c[0] = sensed_concentration;

        if (agents[agent_id].dc_observations<MAX_DC_OBSERVATIONS){
          //not enough: keep adding
          //agents[agent_id].dc[agents[agent_id].dc_observations] = agents[agent_id].c[0] - agents[agent_id].c[1];
          agents[agent_id].dc_observations++;
          } else {
            //enough. cycle through
            	for(int i=0; i<MAX_DC_OBSERVATIONS-1; i++){
            		agents[agent_id].dc[i] = agents[agent_id].dc[i+1];
        		}
            }
        agents[agent_id].dc[agents[agent_id].dc_observations-1] = agents[agent_id].c[0] - agents[agent_id].c[1];
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

__device__ float ztest(float* data, int N_, float x){
  float sum=0.0f;
  float sumsq = 0.0f, p=0.0;
  for(int i=0; i<N_;i++){
    float v=data[i];
    sum+=v;
    sumsq+= v*v;
    }
  float mean = sum/N_;
  float std = sqrtf(sumsq/N_-mean*mean);
  float z = (x-mean)/std;
  p = 0.5f * erfcf(z*0.70710678f);
  return p;
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
    bool leave=false;
    //printf("test\n");
    //printf("state duration %d\n", agents[agent_id].state_duration);
    if(agents[agent_id].state_duration > 1){
      agents[agent_id].previous_state = agents[agent_id].state;
      agents[agent_id].state = agents[agent_id].state; //keep the same state
      agents[agent_id].state_duration -= 1;
        //return; //don't update state if duration not over ->
      // for chemotaxis, duration is just a dummy; allowing self-transitions
    } /*else{
        leave = true;
    }*/ //

    curandState local_rng = rng_states[agent_id];

    int agent_state = agents[agent_id].state;
    float dc = agents[agent_id].c[0] - agents[agent_id].c[1]; //current - previous
    float* transition_matrix_chemotaxis;
    float default_chemotaxis_transition_matrix[9] = {0.68f, 0.10f, 0.22f, 0.03f, 0.27f, 0.7f, 0.01f, 0.02f, 0.98f};
    float p[N_STATES];
    float p_sum = 0.0f;

    //default transition matrix values
    for (int i=0; i<N_STATES; i++){
      p[i] = default_chemotaxis_transition_matrix[agents[agent_id].state * N_STATES + i];
      p_sum += p[i];
     }
	float p_leave = 0.0f; //given by L1


    if(dc>0){
        int n_obs = agents[agent_id].dc_observations;
        if(n_obs > 3) {
            float p_value = ztest(agents[agent_id].dc, n_obs-1, dc);
            switch (agents[agent_id].state) {
                case 0:
                    p[2] = chemotaxis_params_d.a_rev_run * (1.0f - p_value);
                    p[1] = chemotaxis_params_d.a_rev_turn * p_value;
                    p[0] = fmaxf(0.0f, 1.0f - p[1] - p[2]);
                    break;
                case 1:
                    p[2] = chemotaxis_params_d.a_turn_run * (1.0f-p_value);
                    p[0] = chemotaxis_params_d.a_turn_rev * p_value;
                    p[1] = fmaxf(0.0f, 1.0f - p[0] - p[2]);
                    break;
                case 2:
                    p[0] = chemotaxis_params_d.a_run_rev * p_value;
                    p[1] = chemotaxis_params_d.a_run_turn * p_value;
                    p[2] = fmaxf(0.0f, 1.0f - p[0] - p[1]);
                    break;
            }
        }
        p_sum = p[0]+p[1]+p[2];
    } else if(dc<0) {
        switch(agents[agent_id].state){
          case 0:
                p[2] = chemotaxis_params_d.p_rev_run_minus;
                p[1] = chemotaxis_params_d.p_rev_turn_minus;
                p[0] = fmaxf(0.0f, 1.0f - p[1] - p[2]);
                break;
              case 1:
                p[2] = chemotaxis_params_d.p_turn_run_minus;
                p[0] = chemotaxis_params_d.p_turn_rev_minus;
                p[1] = fmaxf(0.0f, 1.0f - p[0] - p[2]);
                break;
              case 2:
                p[0] = chemotaxis_params_d.p_run_rev_minus;
                p[1] = chemotaxis_params_d.p_run_turn_minus;
                p[2] = fmaxf(0.0f, 1.0f - p[0] - p[1]);
                break;
        }
        p_sum = p[0]+p[1]+p[2];
    }

	if(p_sum>0.0f){
        for (int i = 0; i < N_STATES; i++){
            p[i] /= p_sum;
        }
    }
    //transition condition: L1 or state duration expiration
    //if(leave){
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

		agents[agent_id].state_duration = max(new_duration, 1); //at least 1 timestep in the new state

      	BehaviorDistribution state = d_behavior_distributions[next_state];
      	agents[agent_id].p_same_sign = state.p_same_sign;


		agents[agent_id].initial_state_duration = agents[agent_id].state_duration;
	//}
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
