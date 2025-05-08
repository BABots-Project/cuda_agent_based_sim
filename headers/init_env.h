//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_INIT_ENV_H
#define UNTITLED_INIT_ENV_H
#include <cuda_runtime.h>
#include <random>

#include "beta_sampling.h"
#include "../include/json.hpp"
using json = nlohmann::json;

struct State {
    float angle_kappa, angle_mu, speed_alpha, speed_beta, speed_scale, duration_alpha, duration_beta, duration_scale, probability_m, probability_q;
    float speed_loc, duration_loc;
    float max_speed, min_speed;
    float angle_change_mix;
    int breaking_point; //behavioural switch to go to longer-tailed duration
    float duration_scale1, duration_scale2; // duration scale parameters before and after the switch
    float transition_likelihood[N_STATES]; // overall likelihood to transition from state i to j s.t. if i==j it's 0
    int angle_mu_sign = 1; //the sign for the direction of movement, either -1 (left) or 1 (right)
};

struct Agent {
    float x, y, angle, speed, previous_potential, cumulative_potential;  // Position in 2D space
    int state;  // State of the agent: -1 stopped, 0 moving, 1 pirouette
    int is_agent_in_target_area;
    int first_timestep_in_target_area, steps_in_target_area;
    int substate, previous_substate;
    bool is_exploring;
    int state_duration;
    int last_encounter_with_food;
};

struct Parameters{
    Parameters() :
            kappas{0.0f},
            mus{0.0f},
            sigmas{0.0f},
            scales{0.0f}
    {
        // Explicitly zero out parameters
        for(int i = 0; i < N_STATES * WORM_COUNT; i++) {
            kappas[i] = 0.0f;
            mus[i] = 0.0f;
            sigmas[i] = 0.0f;
            scales[i] = 0.0f;
        }
    }

    float kappas[N_STATES * WORM_COUNT];
    float mus[N_STATES * WORM_COUNT];
    float sigmas[N_STATES * WORM_COUNT];
    float scales[N_STATES * WORM_COUNT];
    float probabilities[N_STATES * WORM_COUNT];

};

struct ExplorationState{
    ExplorationState() :
            id(-1),  // Initialize to an invalid state
            speed_scale(0.0f),
            speed_spread(0.0f),
            angle_mu(0.0f),
            angle_kappa(0.0f),
            duration_mu(0.0f),
            duration_sigma(0.0f),
            timesteps_in_state(0),
            duration(0),
            max_duration(0),
            angle_mu_sign(1)
    {
        // Explicitly zero out probabilities
        for(int i = 0; i < N_STATES; i++) {
            probabilities[i] = 0.0f;
        }
    }
    int id, timesteps_in_state;
    int duration, max_duration;
    float speed_scale, speed_spread;
    float angle_mu, angle_kappa;
    float duration_mu, duration_sigma;
    float probabilities[N_STATES];
    int angle_mu_sign;
};


float sample_from_exponential(float lambda){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> d(lambda);
    float sample = d(gen);
    /*while(sample>1.0f){
        sample = d(gen);
    }*/
    return sample;
}

float get_acceptable_white_noise(float mean, float stddev, float bound=1.0f){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);
    float white_noise;

    white_noise = (float) dist(gen);
    while(abs(white_noise)>bound || white_noise<0){
        white_noise = (float) dist(gen);
    }
    return white_noise;
}

void populate_plausible_states(State* source, State* target) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine

    for (int i = 0; i < WORM_COUNT; i++) {
        for (int j = 0; j < N_STATES; j++) {
            int idx = i * N_STATES + j;

            auto uniform = [&](float base, float maximum_value = -1.0f, float minimum_value = -1.0f) {
                return base;
                float multiplier = 1.0f;

                if (base == -1.0f) {
                    return -1.0f;
                }

                float upper_bound = (maximum_value < 0) ? base * multiplier : maximum_value;
                float lower_bound = (minimum_value < 0) ? base / multiplier : minimum_value;

                if (base < 0) {
                    std::swap(lower_bound, upper_bound);
                }

                std::uniform_real_distribution<float> dist(lower_bound, upper_bound);
                float res = dist(gen);

                // Add noise if res is very close to 0
                /*if (std::abs(res) < 1e-5f) {
                    std::uniform_real_distribution<float> noise_dist(-1e-3f, 1e-3f);  // Define new distribution
                    res += noise_dist(gen);  // Sample noise separately
                }*/

                return res;
            };


            target[idx].angle_kappa = uniform(source[j].angle_kappa);
            target[idx].angle_mu = uniform(source[j].angle_mu);
            target[idx].angle_change_mix = uniform(source[j].angle_change_mix);
            target[idx].speed_alpha = uniform(source[j].speed_alpha);
            target[idx].speed_beta = uniform(source[j].speed_beta);
            //target[idx].speed_scale = uniform(source[j].speed_scale);//, MAX_ALLOWED_SPEED*1e3);
            //target[idx].speed_loc = uniform(source[j].speed_loc);//, MAX_ALLOWED_SPEED*1e3);
            target[idx].duration_alpha = uniform(source[j].duration_alpha);
            target[idx].duration_beta = uniform(source[j].duration_beta);
            target[idx].duration_scale = uniform(source[j].duration_scale);//, N_STEPS/2);
            target[idx].duration_loc = uniform(source[j].duration_loc);//, N_STEPS/2);
            target[idx].probability_m = uniform(source[j].probability_m);
            target[idx].probability_q = uniform(source[j].probability_q);
            target[idx].max_speed = uniform(source[j].max_speed);//, MAX_ALLOWED_SPEED*1e3, target[idx].speed_scale * 1e-3);
            target[idx].min_speed = uniform(source[j].min_speed);//,, MAX_ALLOWED_SPEED* 1e3);
            /*while ((target[idx].max_speed - target[idx].min_speed)/target[idx].max_speed < 0.5) {
                target[idx].min_speed = 0.0f;
                target[idx].max_speed = uniform(source[j].max_speed);//,, MAX_ALLOWED_SPEED* 1e3);
            }*/
            target[idx].breaking_point = uniform(source[j].breaking_point);//,, N_STEPS/2); // Assuming it remains unchanged
            target[idx].duration_scale1 = uniform(source[j].duration_scale1);//,, N_STEPS/2);
            target[idx].duration_scale2 = uniform(source[j].duration_scale2);//,, N_STEPS/2);
           /* printf("State %d has speed_alpha = %f, speed_beta = %f, speed_scale = %f, max_speed = %f, min_speed = %f\n", j,  target[idx].speed_alpha,  target[idx].speed_beta,  target[idx].speed_scale, target[idx].max_speed, target[idx].min_speed);
            printf("\tduration_alpha = %f, duration_beta = %f, duration_scale = %f\n", target[idx].duration_alpha, target[idx].duration_beta, target[idx].duration_scale);
            printf("\tbreaking_point=%d, duration_scale1=%f, duration_scale2=%f\n", target[idx].breaking_point, target[idx].duration_scale1, target[idx].duration_scale2);
*/
            // Transition likelihood matrix
            for (int k = 0; k < N_STATES; k++) {
                target[idx].transition_likelihood[k] = (j != k) ? uniform(source[j].transition_likelihood[k]) : 0.0f;
  //              printf("\t\ttransition to %d: %f\n", k, target[idx].transition_likelihood[k]);
            }
            std::uniform_real_distribution<float> dist(0, 1);
            float res = dist(gen);
            if (res > 0.5) {
                target[idx].angle_mu_sign = -1;
            } else {
                target[idx].angle_mu_sign = 1;
            }
            //target[idx].angle_mu_sign = 1;
        }
    }
}

void load_state_data(State* states, const char* filename) {
    printf("parsing json\n");
    std::ifstream file(filename);
    json data = json::parse(file);

    for (int i = 0; i < N_STATES; i++) {
        states[i].angle_kappa = data[state_ids[i]]["angle"]["kappa"];
        states[i].angle_mu = data[state_ids[i]]["angle"]["mu"];
        states[i].angle_change_mix = data[state_ids[i]]["angle"]["mix"];
        states[i].speed_alpha = data[state_ids[i]]["speed"]["alpha"];
        states[i].speed_beta = data[state_ids[i]]["speed"]["beta"];
        //states[i].speed_scale = data[state_ids[i]]["speed"]["scale"];
        if (states[i].speed_alpha == -1.0f) {
            states[i].speed_alpha = 1.0f;
        }
        if (states[i].speed_beta == -1.0f) {
            states[i].speed_beta = 1.0f;
        }
        /*if (states[i].speed_scale == -1.0f) {
            states[i].speed_scale = MAX_ALLOWED_SPEED;
        } /*else {
            states[i].speed_scale *= 1e-3;
        }*/
        states[i].max_speed = data[state_ids[i]]["speed"]["max_value"];
        states[i].min_speed = data[state_ids[i]]["speed"]["min_value"];
        //states[i].speed_loc =  data[state_ids[i]]["speed"]["loc"];
        /*if (states[i].speed_loc == -1.0f) {
            states[i].speed_loc = 0.0f;
        } /*else {
            states[i].speed_loc *= 1e-3;
        }*/
        states[i].probability_m = data[state_ids[i]]["probability"]["m"];
        states[i].probability_q = data[state_ids[i]]["probability"]["q"];
        if (states[i].max_speed == -1.0f) {
            states[i].max_speed = MAX_ALLOWED_SPEED;
        }
        /*else {
            states[i].max_speed =  states[i].max_speed * 1e-3;
        }*/
        if (states[i].min_speed == -1.0f) {
            states[i].min_speed = 0.0f;
        } /*else {
            states[i].min_speed *= 1e-3;
        }*/
        if ((states[i].max_speed - states[i].min_speed) / states[i].max_speed < 0.5) {
            states[i].min_speed = 0.0f;
        }
        //printf("State %d has speed_alpha = %f, speed_beta = %f, speed_scale = %f, max_speed = %f, min_speed = %f\n", i,  states[i].speed_alpha,  states[i].speed_beta,  states[i].speed_scale, states[i].max_speed, states[i].min_speed);
        states[i].duration_alpha = data[state_ids[i]]["duration"]["alpha"];
        states[i].duration_beta = data[state_ids[i]]["duration"]["beta"];
        states[i].duration_scale = data[state_ids[i]]["duration"]["scale"];
        states[i].duration_loc = data[state_ids[i]]["duration"]["loc"];
        /*if (states[i].duration_scale!=-1.0f) {
            states[i].duration_scale/=4.0f;
        }*/
        printf("\tduration_alpha = %f, duration_beta = %f, duration_scale = %f, duration_loc = %f\n", states[i].duration_alpha, states[i].duration_beta, states[i].duration_scale, states[i].duration_loc);
        states[i].breaking_point = data[state_ids[i]]["duration"]["breaking_point"];
        /*if (states[i].breaking_point!=-1.0f) {
            states[i].breaking_point /= 4.0f;
        }*/
        states[i].duration_scale1 = data[state_ids[i]]["duration"]["scale1"];
        /*if (states[i].duration_scale1!=-1.0f) {
            states[i].duration_scale1/=4.0f;
        }*/
        states[i].duration_scale2 = data[state_ids[i]]["duration"]["scale2"];
        /*if (states[i].duration_scale2!=-1.0f) {
            states[i].duration_scale2/=4.0f;
        }*/

        for (int j=0; j<N_STATES; j++) {
            if (i==j) {
                states[i].transition_likelihood[j]=0.0f;
            } else {
                states[i].transition_likelihood[j]=data[state_ids[i]]["transition_likelihood"][state_ids[j]];
            }
        }

    }
    file.close();
}

void load_statistical_data_parameters(Parameters *params, const char *filename) {
    std::ifstream file(filename);
    json data = json::parse(file);
    //the file should be formatted as follows:
    //for each worm, an ID, then inside [ID] there is
    //<state_name>:
    //              probabilities
    //              mu
    //              kappa
    //              scale
    //              sigma
    //              duration_mu
    //              duration_sigma
    for (int i = 0; i < WORM_COUNT; i++) {
        std::string agent_id = std::to_string(i);
        const auto& agent_params = data[agent_id];
        for (int j = 0; j < N_STATES; j++) {
            for (int k = 0; k < N_STATES; k++) {
                params->probabilities[i*N_STATES*N_STATES+j*N_STATES+k] = agent_params["probabilities"][j*N_STATES+k];
            }
            params->kappas[i*N_STATES+j] = agent_params["kappa"];
            params->mus[i*N_STATES+j] = agent_params["mu"];
            params->sigmas[i*N_STATES+j] = agent_params["sigma"];
            params->scales[i*N_STATES+j] = agent_params["scale"];
        }
    }
    file.close();

}

void loadSpeedParameters(Parameters* params, const char* filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);
    for(int i=0; i<WORM_COUNT; i++) {
        for (int j = 0; j < N_STATES; j++) {
            int base_idx = j * 2;
            params->scales[i*N_STATES+j] = data[base_idx].get<float>();
            params->sigmas[i*N_STATES+j] = data[base_idx + 1].get<float>();
        }
    }
    file.close();
}

void loadBatchSingleAgentParameters(Parameters* params, const char* filename, int id){
    //does the same as the others, but puts only the parameters of agent id in the params struct
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);

    // Convert agent ID to string to access its array in JSON
    std::string agent_id = std::to_string(id);
    const auto& agent_params = data[agent_id];
    for(int i=0; i<WORM_COUNT; i++){
        for(int j=0; j<N_STATES; j++){
            int base_idx = j*2; // Each state has Mu and Kappa (2 values)
            params->mus[i*N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i*N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }
    }
    file.close();
}

void loadOptimisedParametersSingleAgent(Parameters* params, const char* filename, int id){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    json data = json::parse(file);
    std::string agent_id = std::to_string(id);
    const auto& agent_params = data[agent_id];
    for(int i=0; i<WORM_COUNT; i++){
        for(int j=0; j<N_STATES; j++){
            int base_idx = j*2; // Each state has Mu and Kappa (2 values)
            params->mus[i*N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i*N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }
    }
    file.close();
}

void loadOptimisedParameters13Agents(Parameters* params, const char* filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Parse the JSON file
    json data = json::parse(file);

    for (int i = 0; i < WORM_COUNT; i++) {
        // Convert agent ID to string to access its array in JSON
        std::string agent_id = std::to_string(i);

        if (!data.contains(agent_id)) {
            std::cerr << "Error: Agent ID " << agent_id << " not found in JSON" << std::endl;
            continue;
        }

        // Access the flat array of parameters for the current agent
        const auto& agent_params = data[agent_id];

        for (int j = 0; j < N_STATES; j++) {
            int base_idx = j * 2; // Each state has Mu and Kappa (2 values)
            params->mus[i * N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i * N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }
    }

    file.close();
}



void loadParameters(Parameters* params, const char* filename, bool load_times=true) {
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);
    //printf("Data: %s\n", data.dump().c_str());
    for (int j = 0; j < N_STATES; j++) {
        int base_idx = j * 2;
        params->mus[j] = data[base_idx].get<float>();
        params->kappas[j] = data[base_idx + 1].get<float>();
    }
    file.close();
}

void set_probabilities(ExplorationState* state, int timestep){
    if (state == nullptr) {
        std::cerr << "Error: Null state pointer" << std::endl;
        return;
    }

    // Ensure id is set before using it
    if (state->id < 0 || state->id >= N_STATES) {
        std::cerr << "Error: Invalid state ID" << std::endl;
        return;
    }
    //float probability_stddev = 0.01f;
    float negative_correlation =     0.1f;
    float positive_correlation =     0.9f;
    float non_correlated =           0.5f;
    float tau=38.0f;//38.0f;
    float pirouette_probability =   fmaxf((-0.002f *((float) timestep) + 2.5f)/tau, 0.25f/tau);// +
    ;//get_acceptable_white_noise(0.0f, probability_stddev);     //2.5 pirouettes per 2 minutes
    float omega_probability =       1.25f/tau;//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);;                                           //1.25 omega turn per 2 minutes
    float reverse_probability =     fmaxf((-0.001f *((float) timestep)+ 1.6f)/tau, 0.5f/tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);    //1.5 reversals per 2 minutes
    float pause_probability =       0.4f/tau;//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                              //0.4 pauses per 2 minutes
    float loop_probability =        0.25f/(tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                           //0.25 loops per 2 minutes
    float arc_probability =         0.75f/(tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                            //0.75 arcs per 2 minutes
    float line_probability =        fmaxf((-0.001f*((float) timestep)+ 2.5f)/(tau), 0.5f/(tau));//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);    //2.5 lines per 2 minutes


    switch(state->id) {
        case 0: {
            /*if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(80.0f, 1.0f, 200.0f);
            }*/
            //state->probabilities[0] = positive_correlation * loop_probability;//sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = negative_correlation * pirouette_probability;
            state->probabilities[4] = positive_correlation * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            //state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        case 1: {
            /*if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(50.0f, 10.0f, 120.0f);
            }*/
            //state->probabilities[1] = positive_correlation * arc_probability; //sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = positive_correlation * omega_probability;
            state->probabilities[5] = negative_correlation * reverse_probability;
            //state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        case 2: {
            /*
            if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(30.0f, 5.0f, 60.0f);
            }*/
            //state->probabilities[2] = positive_correlation * line_probability;// sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = negative_correlation * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            //state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        default: {
            state->probabilities[0] = non_correlated * loop_probability;
            state->probabilities[1] = non_correlated * arc_probability;
            state->probabilities[2] = non_correlated * line_probability;
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = non_correlated * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            //state->probabilities[6] = non_correlated * pause_probability;
            //avoid self loops for now
            state->probabilities[state->id] = 0.0f;
            break;
        }
    }

    // Normalize probabilities
    float total_prob = 0.0f;
    for (int j = 0; j < N_STATES; j++) {
        //check for negative values, set to 0
        if(state->probabilities[j] < 0){
            state->probabilities[j] = 0.0f;
//printf("Negative probability from state %d to state %d at time step %d\n", state->id, j, timestep);
        }/* else if (state->probabilities[j] > 1.0f) {
            state->probabilities[j] = 1.0f;
        }*/
        total_prob += state->probabilities[j];
    }

    if (total_prob > 0.0f) {
        for (int j = 0; j < N_STATES; j++) {
            state->probabilities[j] /= total_prob;
            state->angle_mu_sign = 1;
        }
    }
}

int get_duration(float mu, float sigma, int upper_bound){
    std::mt19937 engine; // uniform random bit engine

    // seed the URBG
    std::random_device dev{};
    engine.seed(dev());
    std::lognormal_distribution<double> dist(mu, sigma);
    int duration = (int) dist(engine);
    int max_retries = 50;
    while (duration < 0 || duration > upper_bound) {
        printf("duration %f\n", dist(engine));
        duration = (int) dist(engine);
        max_retries -= 1;
        if (max_retries<=0) break;
    }
    if (max_retries <= 0) {duration=-1;}
    return duration;
}


int initProbabilitiesWithParams(ExplorationState* states, Parameters* params) {
    //printf("Initializing probabilities\n");

    for (int i = 0; i < WORM_COUNT; i++) {
        //printf("Worm id: %d\n", i);
        for (int j = 0; j < N_STATES; j++) {
            //printf("State id: %d\n", j);
            // Explicitly initialize each exploration state
            states[i*N_STATES + j] = ExplorationState();
            states[i*N_STATES + j].id = j;

            states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
            states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
            states[i*N_STATES + j].duration = 0;


            states[i*N_STATES + j].angle_mu = params->mus[i*N_STATES + j];
            states[i*N_STATES + j].angle_kappa = params->kappas[i*N_STATES + j];
            states[i*N_STATES + j].speed_scale = params->scales[i*N_STATES + j];
            states[i*N_STATES + j].speed_spread = params->sigmas[i*N_STATES + j];


        }
        //printf("Setting up durations\n");
        // Set probabilities for each state after initialization
        for (int j = 0; j < N_STATES; j++) {
            set_probabilities(&states[i*N_STATES + j], 0);
        }
    }
    return 0;
}


void initProbabilities(ExplorationState* states) {
    printf("Initializing probabilities\n");

    for (int i = 0; i < WORM_COUNT; i++) {

        for (int j = 0; j < N_STATES; j++) {

            // Explicitly initialize each exploration state
            states[i*N_STATES + j] = ExplorationState();
            states[i*N_STATES + j].id = j;
            states[i*N_STATES + j].angle_mu_sign = 1;
            if(j<3 || j==5){ //crawling states
                states[i*N_STATES + j].speed_scale = OFF_FOOD_SPEED_SCALE_FAST;
                states[i*N_STATES + j].speed_spread = OFF_FOOD_SPEED_SHAPE_FAST;

            }else{ //turning states
                states[i*N_STATES + j].speed_scale = OFF_FOOD_SPEED_SCALE_SLOW;
                states[i*N_STATES + j].speed_spread = OFF_FOOD_SPEED_SHAPE_SLOW;
            }
            states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
            states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
            states[i*N_STATES + j].duration = 0;

            //states[i*N_STATES + j].max_duration = 0;
            switch(j){
                case 0:
                    states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 200;
                    states[i*N_STATES + j].angle_mu = M_PI/12;
                    states[i*N_STATES + j].angle_kappa = 15.0f;
                    break;
                case 1:
                    states[i*N_STATES + j].duration_mu = ARC_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = ARC_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 140;
                    states[i*N_STATES + j].angle_mu = M_PI/12;
                    states[i*N_STATES + j].angle_kappa = 10.0f;
                    break;
                case 2:
                    states[i*N_STATES + j].duration_mu = LINE_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = LINE_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 60;
                    states[i*N_STATES + j].angle_mu = 0;
                    states[i*N_STATES + j].angle_kappa = 15.0f;
                    break;
                case 3:
                    states[i*N_STATES + j].angle_mu = 3*M_PI/4;
                    states[i*N_STATES + j].angle_kappa = 2.0f;
                    break;
                case 4:
                    states[i*N_STATES + j].angle_mu = M_PI/2;
                    states[i*N_STATES + j].angle_kappa = 0.5f;
                    break;
                case 5:
                    states[i*N_STATES + j].angle_mu = 0.0f;
                    states[i*N_STATES + j].angle_kappa = 0.75f;
                    break;
                case 6:
                    states[i*N_STATES + j].angle_mu = 0.0f;
                    states[i*N_STATES + j].angle_kappa = 2.0f;
                    break;
            }
        }

        // Set probabilities for each state after initialization
        for (int j = 0; j < N_STATES; j++) {
            if(j<3) {
                states[i * N_STATES + j].duration = get_duration(states[i * N_STATES + j].duration_mu, states[i * N_STATES + j].duration_sigma, states[i * N_STATES + j].max_duration);
                //  printf("Duration for state %d: %d\n", j, states[i * N_STATES + j].duration);
            }
            set_probabilities(&states[i*N_STATES + j], 0);
        }
    }

}

void updateProbabilities(ExplorationState* states, int timestep){
    for (int i = 0; i < WORM_COUNT; i++) {
        for (int j = 0; j < N_STATES; j++) {
            set_probabilities(&states[i*N_STATES + j], timestep);
        }
    }
}


// CUDA kernel to initialize the position of each agent
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed, int worm_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        curand_init(seed, id, 0, &states[id]);
        if (ENABLE_RANDOM_INITIAL_POSITIONS) {
            agents[id].x = curand_uniform(&states[id]) * WIDTH;
            agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        } else {
            //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
           agents[id].x = WIDTH / 2 - INITIAL_AREA_NUMBER_OF_CELLS/2 * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
           agents[id].y = HEIGHT / 2 - INITIAL_AREA_NUMBER_OF_CELLS/2  * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
            //agents[id].x = 43.000000;
            //agents[id].y = 30.000000;
        }
        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = 0.0f;
        float generated_value = curand_uniform(&states[id]);
        agents[id].state = static_cast<int>(generated_value*(N_STATES));
        agents[id].previous_potential = 0.0f;
        agents[id].cumulative_potential = 0.0f;
        agents[id].is_agent_in_target_area = 0;
        agents[id].first_timestep_in_target_area = -1;
        agents[id].steps_in_target_area = 0;
        agents[id].is_exploring = true;
        agents[id].substate = static_cast<int>(curand_uniform(&states[id]) * N_STATES);
        agents[id].previous_substate = 0;
        agents[id].state_duration = -1;
        agents[id].last_encounter_with_food = 0;
    }
}

// CUDA kernel to initialize the chemical grid concentration
__global__ void initGrid(float* grid, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the square in the middle of the grid with length 20
        if (i >= 3 * N / 4 -  TARGET_AREA_SIDE_LENGTH/2 && i < 3 * N / 4 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
        //if (i >=N / 2 - TARGET_AREA_SIDE_LENGTH/2 && i <  N / 2 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
            grid[i * N + j] = MAX_CONCENTRATION * (1.0f + curand_normal(&states[i*N+j]));
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}

// CUDA kernel to initialize the chemical grid concentration in an approximated circle of radius TARGET_AREA_SIDE_LENGTH/2
__global__ void initGridWithCircle(float* grid, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the circle in the middle of the grid with radius 20
        if ((i - 3 * N / 4) * (i - 3 * N / 4) + (j - N / 2) * (j - N / 2) <= (TARGET_AREA_SIDE_LENGTH / 2) * (TARGET_AREA_SIDE_LENGTH / 2)) {
            grid[i * N + j] = MAX_CONCENTRATION * (1.0f + curand_normal(&states[i*N+j]));
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}


// CUDA kernel to initialize the chemical grid with two squares of chemical placed in the lower left and upper right corners. size 10x10 cells each
__global__ void initGridWithTwoSquares(float* grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        int upper_right_center_x = 3*N/4;
        int upper_right_center_y = 3*N/4;
        int lower_left_center_x = N/4;
        int lower_left_center_y = N/4;
        if ((i >= upper_right_center_x - 5 && i < upper_right_center_x + 5 && j >= upper_right_center_y - 5 && j < upper_right_center_y + 5) || (i >= lower_left_center_x - 5 && i < lower_left_center_x + 5 && j >= lower_left_center_y - 5 && j < lower_left_center_y + 5)) {
            grid[i * N + j] = MAX_CONCENTRATION;
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}


// CUDA kernel to initialize the pheromone grids
__global__ void initAttractiveAndRepulsivePheromoneGrid(float* attractive_pheromone, float* repulsive_pheromone, int* agent_density_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        if(agent_density_grid[i * N + j] == 0){
            attractive_pheromone[i * N + j] = 0.0f;
            repulsive_pheromone[i * N + j] = 0.0f;
        }
        else {
            attractive_pheromone[i * N + j] = ATTRACTANT_PHEROMONE_SECRETION_RATE * ATTRACTANT_PHEROMONE_DECAY_RATE *
                                              (float) agent_density_grid[i * N + j] / (DX * DX);
            repulsive_pheromone[i * N + j] = REPULSIVE_PHEROMONE_SECRETION_RATE * REPULSIVE_PHEROMONE_DECAY_RATE *
                                             (float) agent_density_grid[i * N + j] / (DX * DX);
        }
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * N + j]);
    }
}

__global__ void initRepulsivePheromoneGrid(float* repulsive_pheromone, int* agent_density_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        if(agent_density_grid[i * N + j] == 0){
            repulsive_pheromone[i * N + j] = 0.0f;
        }
        else {
            repulsive_pheromone[i * N + j] = REPULSIVE_PHEROMONE_SECRETION_RATE * REPULSIVE_PHEROMONE_DECAY_RATE *
                                             (float) agent_density_grid[i * N + j] / (DX * DX);
        }
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * N + j]);
    }
}

//CUDA kernel to initialise the agent count grid
__global__ void initAgentDensityGrid(int* agent_count_grid, Agent* agents, int worm_count){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        agent_count_grid[i * N + j] = 0;
        for (int k = 0; k < worm_count; ++k) {
            int agent_x = (int)(agents[k].x / DX);
            int agent_y = (int)(agents[k].y / DX);
            if (agent_x == i && agent_y == j) {
                // printf("Agent at (%d, %d)\n", i, j);
                agent_count_grid[i * N + j] += 1;
            }
        }
    }
}


void initBacterialLawnSquareHost(float * bacterial_lawn, int x_centers[N_FOOD_SPOTS], int y_centers[N_FOOD_SPOTS]) {
    for (int k=0; k<N_FOOD_SPOTS; k++) {
        float density = bacterial_densities[k];
        float x_center = x_centers[k];
        float y_center = y_centers[k];
        for (int i=x_center-SPOT_SIZE/2; i<x_center+SPOT_SIZE/2; i++) {
            for (int j=y_center-SPOT_SIZE/2; j<y_center+SPOT_SIZE/2; j++) {
                bacterial_lawn[i*N+j] = density;
            }
        }
    }
}

#endif //UNTITLED_INIT_ENV_H
