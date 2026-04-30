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

struct JointTable {
    int    n;         // number of observations
    float* obs;       // device ptr: interleaved [speed0, angle0, speed1, angle1, ...]
    float* prob;      // device ptr: alias prob array, length n
    int*   alias;     // device ptr: alias index array, length n
};

struct StateParams {
    // BetaPrime duration params
    float bp_alpha;
    float bp_beta;
    float bp_scale;

    // Conditional joint tables (one per duration level)
    int         n_durations;
    int*        durations;   // device ptr: sorted int array, length n_durations
    JointTable* tables;      // device ptr: JointTable array, length n_durations
};

// Host-side helper (mirrors device layout, owns host memory)
struct StateParamsHost {
    float bp_alpha, bp_beta, bp_scale;
    int                       n_durations;
    std::vector<int>          durations;
    std::vector<JointTable>   tables;      // each table's ptrs are HOST ptrs here
    // flat storage for obs/prob/alias per table
    std::vector<std::vector<float>> obs_data;
    std::vector<std::vector<float>> prob_data;
    std::vector<std::vector<int>>   alias_data;
};

// ---- helpers ---------------------------------------------------

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
}

template<typename T>
static T* device_alloc_copy(const std::vector<T>& src) {
    T* d_ptr = nullptr;
    cuda_check(cudaMalloc(&d_ptr, src.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_ptr, src.data(),
                          src.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return d_ptr;
}

// ---- load BetaPrime params from JSON ---------------------------

static void load_betaprime(const char* path,
                            int n_states,
                            std::vector<StateParamsHost>& host_params) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    json j = json::parse(f);

    host_params.resize(n_states);
    for (auto it = j.items().begin(); it != j.items().end(); ++it) {
    const std::string& key = it.key();
    const json& val = it.value();

    int state = std::stoi(key);
    if (state < 0 || state >= n_states)
        throw std::runtime_error("State index out of range: " + key);

    host_params[state].bp_alpha = val["alpha"].get<float>();
    host_params[state].bp_beta  = val["beta"].get<float>();
    host_params[state].bp_scale = val["scale"].get<float>();
}
}

// ---- load joint distributions from JSON ------------------------

static void load_joint(const char* path,
                       std::vector<StateParamsHost>& host_params) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);
    json j = json::parse(f);

    for (auto it = j.items().begin(); it != j.items().end(); ++it) {
        const std::string& s_key = it.key();
        const json& dur_map = it.value();

        int state = std::stoi(s_key);
        StateParamsHost& sp = host_params[state];

        // collect and sort duration levels
        std::vector<int> dur_keys;
        for (auto it2 = dur_map.items().begin(); it2 != dur_map.items().end(); ++it2) {
            const std::string& d_key = it2.key();
            dur_keys.push_back(std::stoi(d_key));
        }
        std::sort(dur_keys.begin(), dur_keys.end());

        sp.n_durations = (int)dur_keys.size();
        sp.durations   = dur_keys;
        sp.tables.resize(sp.n_durations);
        sp.obs_data.resize(sp.n_durations);
        sp.prob_data.resize(sp.n_durations);
        sp.alias_data.resize(sp.n_durations);

        for (int i = 0; i < sp.n_durations; ++i) {
            std::string d_key = std::to_string(dur_keys[i]);
            const json& entry = dur_map[d_key];
            int n = entry["n"].get<int>();

            // obs: array of [speed, angle] pairs → flatten to float*
            sp.obs_data[i].reserve(n * 2);
            for (auto it3 = entry["obs"].begin(); it3 != entry["obs"].end(); ++it3) {
                const auto& pair = *it3;
                for (auto it4 = pair.begin(); it4 != pair.end(); ++it4) {
                    float v = *it4;
                    sp.obs_data[i].push_back(v);
                }
            }

            sp.prob_data[i]  = entry["prob"].get<std::vector<float>>();
            sp.alias_data[i] = entry["alias"].get<std::vector<int>>();

            sp.tables[i].n     = n;
            sp.tables[i].obs   = nullptr;  // filled in upload step
            sp.tables[i].prob  = nullptr;
            sp.tables[i].alias = nullptr;
        }
    }
}

// ---- upload one state to device --------------------------------

static void upload_state(const StateParamsHost& sp, StateParams& out_d) {
    out_d.bp_alpha = sp.bp_alpha;
    out_d.bp_beta  = sp.bp_beta;
    out_d.bp_scale = sp.bp_scale;
    out_d.n_durations = sp.n_durations;

    // 1. leaf arrays: durations
    out_d.durations = device_alloc_copy(sp.durations);

    // 2. for each table: upload obs/prob/alias, build a host-side JointTable
    //    with device pointers, then upload the array of those structs
    std::vector<JointTable> tables_with_dptrs(sp.n_durations);
    for (int i = 0; i < sp.n_durations; ++i) {
        tables_with_dptrs[i].n     = sp.tables[i].n;
        tables_with_dptrs[i].obs   = device_alloc_copy(sp.obs_data[i]);
        tables_with_dptrs[i].prob  = device_alloc_copy(sp.prob_data[i]);
        tables_with_dptrs[i].alias = device_alloc_copy(sp.alias_data[i]);
    }

    // 3. now copy the JointTable structs (which contain device ptrs) to device
    out_d.tables = device_alloc_copy(tables_with_dptrs);
}

// ---- public entry point ----------------------------------------

void load_distributions(const char* betaprime_path,
                         const char* joint_path,
                         int         n_states,
                         StateParams** d_params_out)   // device array
{
    std::vector<StateParamsHost> host_params;

    load_betaprime(betaprime_path, n_states, host_params);
    load_joint(joint_path, host_params);

    // build device StateParams array
    std::vector<StateParams> h_state_params(n_states);
    for (int s = 0; s < n_states; ++s)
        upload_state(host_params[s], h_state_params[s]);

    *d_params_out = device_alloc_copy(h_state_params);
}


struct DurationLognormal
{
    float mu;
    float sigma;
};


struct BehaviorDistribution
{
    int n_speed_bins;
    float* speed_bins;
    float* speed_prob;
    int*   speed_alias;

    int n_roaming_speed_bins;
    float* roaming_speed_bins;
    float* roaming_speed_prob;
    int*   roaming_speed_alias;


    float speed_alpha;
    float speed_mean;

    int n_angle_bins;
    float* angle_bins;
    float* angle_prob;
    int*   angle_alias;

    int roaming_n_angle_bins;
    float* roaming_angle_bins;
    float* roaming_angle_prob;
    int*   roaming_angle_alias;

    int first_n_angle_bins;
    float* first_angle_bins;
    float* first_angle_prob;
    int*   first_angle_alias;

	float angle_alpha;
    float angle_mean;

    int n_angle_bins_persistent;
    float* angle_bins_persistent;
    float* angle_prob_persistent;
    int*   angle_alias_persistent;

    int n_angle_bins_non_persistent;
    float* angle_bins_non_persistent;
    float* angle_prob_non_persistent;
    int*   angle_alias_non_persistent;

    //duration
    int n_duration_bins;
    int* duration_bins;
    float* duration_prob;

    float f_plus;

    float sign_concordance;
    float mean_angular_difference;
    float std_angular_difference;
    float p_same_sign;



};

struct TransitionModel
{
    float p_off_food;
    int tau;
    float coeff;
    float intercept;
    float mean, std; //scale params
    int sign;
    float height;
};

struct TransitionFactor{
  float angle_plus, angle_minus;
  float speed_plus, speed_minus;
};

struct TransitionBias{
    float angle_plus, angle_minus;
};

struct PRoam{
  float p_roam;
  int thresh;
  };

__constant__ BehaviorDistribution d_behavior_distributions[N_STATES];
__constant__ TransitionModel d_transition_models[N_STATES*N_STATES];
__constant__ TransitionModel d_exit_models[N_STATES];
__constant__ float odor_x0;
__constant__ float odor_y0;
float h_odor_x0 = WIDTH/2.0f;
float h_odor_y0 = HEIGHT/2.0f;
__constant__ float frequencies[N_STATES];
__constant__ TransitionFactor d_transition_factors[N_STATES*N_STATES];
__constant__ TransitionBias d_transition_biases[N_STATES*N_STATES];
__constant__ DurationLognormal d_duration_lognormals[N_STATES];
__constant__ PRoam d_proam;
__constant__ DurationLognormal d_roaming_lognormal;

struct PRoamHost{
  float p_roam;
  int thresh;
  };

struct DurationLognormalHost
{
    float mu;
    float sigma;
};

struct BehaviorDistributionHost
{
    std::vector<float> speed_bins;
    std::vector<float> speed_prob;
    std::vector<int>   speed_alias;
    float speed_alpha;
    float speed_mean;

    std::vector<float> angle_bins;
    std::vector<float> angle_prob;
    std::vector<int>   angle_alias;

     std::vector<float> roaming_angle_bins;
    std::vector<float> roaming_angle_prob;
    std::vector<int>   roaming_angle_alias;

     std::vector<float> roaming_speed_bins;
    std::vector<float> roaming_speed_prob;
    std::vector<int>   roaming_speed_alias;


    std::vector<float> first_angle_bins;
    std::vector<float> first_angle_prob;
    std::vector<int>   first_angle_alias;
    float angle_alpha;
    float angle_mean;

    std::vector<float> angle_bins_persistent;
    std::vector<float> angle_prob_persistent;
    std::vector<int>   angle_alias_persistent;

    std::vector<float> angle_bins_non_persistent;
    std::vector<float> angle_prob_non_persistent;
    std::vector<int>   angle_alias_non_persistent;

    //duration
    std::vector<int> duration_bins;
    std::vector<float> duration_prob;

    float sign_concordance;
    float f_plus;
    float mean_angular_difference;
    float std_angular_difference;
    float p_same_sign;
};

struct TransitionModelHost
{
    float p_off_food;
    int tau;
    float coeff;
    float intercept;
    float mean, std; //scale params
    int sign;
    float height;
};

struct TransitionBiasHost{
    float angle_plus, angle_minus;
};

struct Agent {
    float x, y, angle, speed;  // Position in 2D space
    float angle_change;
    float previous_angle, previous_speed;
    float previous_mag_angle_change;
    int state;  // State of the agent
    int previous_state;
    float c[100]; // Array to store the last 100 concentration values for the agent
    float dc_int[N_STATES*N_STATES]; // Array to store the dc_int[tau] values for the agent, one for each transition
    float accumulated_dc_tot; // Variable to store the accumulated DC total for the agent
    int angle_sign;
    bool is_persistent;
    int state_duration;
    float p_same_sign;
    int initial_state_duration;
    float kappa;
    float phi;
    float run_omega = 0.0f;
	float run_amp = 0.0f;
    int agent_id;
    int neighbor_count;
    int prev_neighbor_count, delta_neighbor_count;
};

struct TransitionFactorHost{
  float angle_plus, angle_minus;
  float speed_plus, speed_minus;
};


// ── host struct to hold the label array on device ────────────────────────────
struct WormLabelSequence {
    int*  d_labels;     // device pointer
    int   length;       // total number of timesteps
};

// ── minimal JSON parser (no external deps) ───────────────────────────────────
// Expects exactly the format above: a JSON array of integers on one line.
static int* load_json_labels(const char* path, int* out_length) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    // read whole file
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(fsize + 1);
    fread(buf, 1, fsize, f);
    buf[fsize] = '\0';
    fclose(f);

    // count commas to pre-allocate
    int n = 1;
    for (long i = 0; i < fsize; i++) if (buf[i] == ',') n++;

    int* arr = (int*)malloc(n * sizeof(int));
    int  idx = 0;
    char* p  = buf;
    while (*p) {
        // skip until digit or minus
        while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
        if (!*p) break;
        arr[idx++] = (int)strtol(p, &p, 10);
    }
    free(buf);

    *out_length = idx;
    return arr;
}

// ── public loader: fills a WormLabelSequence ─────────────────────────────────
WormLabelSequence load_worm_labels_to_device(char* json_path) {
    WormLabelSequence seq = {NULL, 0};

    int   n_labels  = 0;
    int*  h_labels  = load_json_labels(json_path, &n_labels);
    if (!h_labels) return seq;

    cudaMalloc(&seq.d_labels, n_labels * sizeof(int));
    cudaMemcpy(seq.d_labels, h_labels, n_labels * sizeof(int),
               cudaMemcpyHostToDevice);
    free(h_labels);

    seq.length = n_labels;
    printf("Loaded %d labels from %s\n", n_labels, json_path);
    return seq;
}

void free_worm_labels(WormLabelSequence* seq) {
    if (seq->d_labels) cudaFree(seq->d_labels);
    seq->d_labels = NULL;
    seq->length   = 0;
}


void load_p_roam(PRoamHost* proam, const char* filename){
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }
	json data = json::parse(file);
    proam->p_roam = data["p_roam"].get<float>();
    proam->thresh = data["T_threshold"].get<int>();
}

void upload_proam(PRoamHost* proam){
  PRoam h_gpu_proam;
  h_gpu_proam.p_roam = proam->p_roam;
  h_gpu_proam.thresh = proam->thresh;
  cudaMemcpyToSymbol(d_proam,
                       &h_gpu_proam,
                       sizeof(PRoam));
  }

void load_duration_data(DurationLognormalHost* durations, const char* filename, DurationLognormalHost* h_roaming_duration){
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }
	json data = json::parse(file);
 	for (int i = 0; i <N_STATES; i++)
    {
        durations[i].mu = data[state_ids[i]]["mu"].get<float>();
        durations[i].sigma = data[state_ids[i]]["sigma"].get<float>();
    }
    //h_roaming_duration->mu = data["roam"]["mu"].get<float>();
    //h_roaming_duration->sigma = data["roam"]["sigma"].get<float>();

}

void load_state_data(BehaviorDistributionHost* states, const char* filename, PRoamHost* h_proam)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    json data = json::parse(file);
	printf("Loading state data from %s\n", filename);
    for (int i = 0; i < N_STATES; i++)
    {
        auto& s = states[i];

        auto speed = data[state_ids[i]]["speed"];
        auto angle = data[state_ids[i]]["angle_change"];

        s.speed_bins  = speed["bin_centers"].get<std::vector<float>>();
        s.speed_prob  = speed["prob"].get<std::vector<float>>();
        s.speed_alias = speed["alias"].get<std::vector<int>>();
        s.speed_alpha = data[state_ids[i]]["alpha_speed"].get<float>();
		s.speed_mean = data[state_ids[i]]["mean_speed"].get<float>();

        s.angle_bins  = angle["bin_centers"].get<std::vector<float>>();
        s.angle_prob  = angle["prob"].get<std::vector<float>>();
        s.angle_alias = angle["alias"].get<std::vector<int>>();
        //we use the STATE_MAX_DURATIONS list
        //if the cur state has the max duration above d_proam.thresh, then we load as base probabilities the dwell probabilities
        //and we populate the roaming attributes (which do not exist in the json otherwise)


		s.angle_alpha = data[state_ids[i]]["alpha_angle_change"].get<float>();
		s.angle_mean = data[state_ids[i]]["mean_angle_change"].get<float>();
        s.sign_concordance =0.0f;
       	auto under_mean_angle_change = data[state_ids[i]]["under_mean_angle_change"];
        s.angle_bins_persistent  = under_mean_angle_change["bin_centers"].get<std::vector<float>>();
        s.angle_prob_persistent  = under_mean_angle_change["prob"].get<std::vector<float>>();
        s.angle_alias_persistent = under_mean_angle_change["alias"].get<std::vector<int>>();
        auto above_mean_angle_change = data[state_ids[i]]["above_mean_angle_change"];
    	s.angle_bins_non_persistent  = above_mean_angle_change["bin_centers"].get<std::vector<float>>();
        s.angle_prob_non_persistent  = above_mean_angle_change["prob"].get<std::vector<float>>();
        s.angle_alias_non_persistent = above_mean_angle_change["alias"].get<std::vector<int>>();
        s.f_plus = data[state_ids[i]]["f_plus"].get<float>();

        auto duration = data[state_ids[i]]["duration"];
        s.duration_bins  = duration["values"].get<std::vector<int>>();
        s.duration_prob  = duration["cumprobs"].get<std::vector<float>>();
		auto first_angle_changes = data[state_ids[i]]["first_angle_changes"];
        s.first_angle_bins  = first_angle_changes["bin_centers"].get<std::vector<float>>();
        s.first_angle_prob  = first_angle_changes["prob"].get<std::vector<float>>();
        s.first_angle_alias = first_angle_changes["alias"].get<std::vector<int>>();

        s.mean_angular_difference = data[state_ids[i]]["mean_angular_difference"].get<float>();
        s.std_angular_difference = data[state_ids[i]]["std_angular_difference"].get<float>();
        s.p_same_sign = data[state_ids[i]]["same_sign_prob"].get<float>();
    }
}

void upload_duration_data(DurationLognormalHost* h_durations, DurationLognormalHost* h_roaming_duration)
{
    DurationLognormal h_gpu_durations[N_STATES], h_gpu_roaming_duration;

    for (int i = 0; i < N_STATES; i++)
    {
        h_gpu_durations[i].mu = h_durations[i].mu;
        h_gpu_durations[i].sigma = h_durations[i].sigma;
    }

    cudaMemcpyToSymbol(d_duration_lognormals,
                       h_gpu_durations,
                       sizeof(DurationLognormal)*N_STATES);

    /*h_gpu_roaming_duration.mu = h_roaming_duration->mu;
    h_gpu_roaming_duration.sigma = h_roaming_duration->sigma;
    cudaMemcpyToSymbol(d_roaming_lognormal,
                       &h_gpu_roaming_duration,
                       sizeof(DurationLognormal));*/
}

void upload_distributions(BehaviorDistributionHost* h_states, PRoamHost* h_proam)
{
    BehaviorDistribution h_gpu_states[N_STATES];

    for (int i = 0; i < N_STATES; i++)
    {
        auto& src = h_states[i];
        auto& dst = h_gpu_states[i];

        int ns = src.speed_bins.size();
        int na = src.angle_bins.size();

        dst.n_speed_bins = ns;
        dst.n_angle_bins = na;

        cudaMalloc(&dst.speed_bins,  ns * sizeof(float));
        cudaMalloc(&dst.speed_prob,  ns * sizeof(float));
        cudaMalloc(&dst.speed_alias, ns * sizeof(int));

        cudaMalloc(&dst.angle_bins,  na * sizeof(float));
        cudaMalloc(&dst.angle_prob,  na * sizeof(float));
        cudaMalloc(&dst.angle_alias, na * sizeof(int));

        cudaMemcpy(dst.speed_bins,  src.speed_bins.data(),  ns*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.speed_prob,  src.speed_prob.data(),  ns*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.speed_alias, src.speed_alias.data(), ns*sizeof(int),   cudaMemcpyHostToDevice);
		dst.speed_alpha = src.speed_alpha;
        dst.speed_mean = src.speed_mean;
        cudaMemcpy(dst.angle_bins,  src.angle_bins.data(),  na*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.angle_prob,  src.angle_prob.data(),  na*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.angle_alias, src.angle_alias.data(), na*sizeof(int),   cudaMemcpyHostToDevice);
		dst.angle_alpha = src.angle_alpha;
        dst.angle_mean = src.angle_mean;
    	dst.sign_concordance = src.sign_concordance;

        if (h_STATE_MAX_DURATIONS[i] > h_proam->thresh){
          int nr = src.roaming_angle_bins.size();
          dst.roaming_n_angle_bins = nr;
          cudaMalloc(&dst.roaming_angle_bins,  nr * sizeof(float));
        cudaMalloc(&dst.roaming_angle_prob,  nr * sizeof(float));
        cudaMalloc(&dst.roaming_angle_alias, nr * sizeof(int));
		cudaMemcpy(dst.roaming_angle_bins,  src.roaming_angle_bins.data(),  nr*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.roaming_angle_prob,  src.roaming_angle_prob.data(),  nr*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.roaming_angle_alias, src.roaming_angle_alias.data(), nr*sizeof(int),   cudaMemcpyHostToDevice);
			int nsr = src.roaming_speed_bins.size();
          dst.n_roaming_speed_bins = nsr;
          cudaMalloc(&dst.roaming_speed_bins,  nsr * sizeof(float));
        cudaMalloc(&dst.roaming_speed_prob,  nsr * sizeof(float));
        cudaMalloc(&dst.roaming_speed_alias, nsr * sizeof(int));
		cudaMemcpy(dst.roaming_speed_bins,  src.roaming_speed_bins.data(),  nsr*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.roaming_speed_prob,  src.roaming_speed_prob.data(),  nsr*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.roaming_speed_alias, src.roaming_speed_alias.data(), nsr*sizeof(int),   cudaMemcpyHostToDevice);


          }

        dst.n_angle_bins_non_persistent = src.angle_bins_non_persistent.size();
        cudaMalloc(&dst.angle_bins_non_persistent,  dst.n_angle_bins_non_persistent * sizeof(float));
        cudaMalloc(&dst.angle_prob_non_persistent,  dst.n_angle_bins_non_persistent * sizeof(float));
        cudaMalloc(&dst.angle_alias_non_persistent, dst.n_angle_bins_non_persistent * sizeof(int));
        cudaMemcpy(dst.angle_bins_non_persistent,  src.angle_bins_non_persistent.data(),  dst.n_angle_bins_non_persistent*sizeof(float), cudaMemcpyHostToDevice);
       	cudaMemcpy(dst.angle_prob_non_persistent,  src.angle_prob_non_persistent.data(),  dst.n_angle_bins_non_persistent*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.angle_alias_non_persistent, src.angle_alias_non_persistent.data(), dst.n_angle_bins_non_persistent*sizeof(int),   cudaMemcpyHostToDevice);

        dst.n_angle_bins_persistent = src.angle_bins_persistent.size();
        cudaMalloc(&dst.angle_bins_persistent,  dst.n_angle_bins_persistent * sizeof(float));
        cudaMalloc(&dst.angle_prob_persistent,  dst.n_angle_bins_persistent * sizeof(float));
        cudaMalloc(&dst.angle_alias_persistent, dst.n_angle_bins_persistent * sizeof(int));
        cudaMemcpy(dst.angle_bins_persistent,  src.angle_bins_persistent.data(),  dst.n_angle_bins_persistent*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.angle_prob_persistent,  src.angle_prob_persistent.data(),  dst.n_angle_bins_persistent*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.angle_alias_persistent, src.angle_alias_persistent.data(), dst.n_angle_bins_persistent*sizeof(int),   cudaMemcpyHostToDevice);

        dst.f_plus = src.f_plus;

        dst.n_duration_bins = src.duration_bins.size();
        cudaMalloc(&dst.duration_bins,  dst.n_duration_bins * sizeof(int));
        cudaMalloc(&dst.duration_prob,  dst.n_duration_bins * sizeof(float));
        cudaMemcpy(dst.duration_bins,  src.duration_bins.data(),  dst.n_duration_bins*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.duration_prob,  src.duration_prob.data(),  dst.n_duration_bins*sizeof(float), cudaMemcpyHostToDevice);

        dst.first_angle_bins  = src.first_angle_bins.data();
        dst.first_angle_prob  = src.first_angle_prob.data();
        dst.first_angle_alias = src.first_angle_alias.data();
        dst.first_n_angle_bins = src.first_angle_bins.size();
        cudaMalloc(&dst.first_angle_bins,  dst.n_angle_bins * sizeof(float));
        cudaMalloc(&dst.first_angle_prob,  dst.n_angle_bins * sizeof(float));
        cudaMalloc(&dst.first_angle_alias, dst.n_angle_bins * sizeof(int));

        cudaMemcpy(dst.first_angle_bins,  src.first_angle_bins.data(),  dst.n_angle_bins*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.first_angle_prob,  src.first_angle_prob.data(),  dst.n_angle_bins*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.first_angle_alias, src.first_angle_alias.data(), dst.n_angle_bins*sizeof(int),   cudaMemcpyHostToDevice);

        dst.mean_angular_difference = src.mean_angular_difference;
        dst.std_angular_difference = src.std_angular_difference;
        dst.p_same_sign = src.p_same_sign;


    }

    cudaMemcpyToSymbol(d_behavior_distributions,
                       h_gpu_states,
                       sizeof(BehaviorDistribution)*N_STATES);
}

void load_exit_data(TransitionModelHost* exit_models, const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }
    json data = json::parse(file);
    for (int i = 0; i < N_STATES; i++)
    {
      	if (i!=2) continue;
        auto& src = data[state_ids[i]];

        exit_models[i].p_off_food = src["p_off_food"].get<float>();
        exit_models[i].coeff      = src["model_coeff"].get<float>();
        exit_models[i].intercept  = src["model_intercept"].get<float>();
        exit_models[i].height     = src["model_height"].get<float>();
    }
}

void load_transition_data(TransitionModelHost* models, const char* filename, float frequencies_host[N_STATES])
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    json data = json::parse(file);

    for (int i = 0; i < N_STATES; i++)
    {
        auto& src_from = data[state_ids[i]];

        //frequencies_host[i] = src_from["frequency"].get<float>();

        for (int j = 0; j < N_STATES; j++)
        {
            auto& src_to = src_from[state_ids[j]];

            int idx = i * N_STATES + j;

            models[idx].p_off_food = src_to["p_off_food"].get<float>();
            models[idx].tau        = src_to["tau"].get<int>();
            models[idx].coeff      = src_to["model_coeff"].get<float>();
            models[idx].intercept  = src_to["model_intercept"].get<float>();
            models[idx].mean       = src_to["mean"].get<float>();
            models[idx].std        = src_to["std"].get<float>();
        	models[idx].sign       = src_to["sign"].get<int>();
            models[idx].height     = src_to["model_height"].get<float>();
        }
    }
}

void load_transition_factors(TransitionFactorHost* factors, const char* filename)
{    std::ifstream file(filename);
  	if (!file.is_open()){
          printf("Could not open %s\n", filename);
        	exit(1);
    }
    json data = json::parse(file);

    for(int i=0; i<N_STATES; i++){
        auto& src_from = data[state_ids[i]];
        for(int j=0; j<N_STATES; j++)
        {
            auto& src_to = src_from[state_ids[j]];
            int idx = i * N_STATES + j;
            factors[idx].angle_plus = src_to["angle_plus"].get<float>();
            factors[idx].angle_minus = src_to["angle_minus"].get<float>();
            factors[idx].speed_plus = src_to["speed_plus"].get<float>();
            factors[idx].speed_minus = src_to["speed_minus"].get<float>();
        }
    }
}

void load_transition_biases(TransitionBiasHost* biases, const char* filename)
{    std::ifstream file(filename);
  	if (!file.is_open()){
          printf("Could not open %s\n", filename);
        	exit(1);
    }
    json data = json::parse(file);

    for(int i=0; i<N_STATES; i++){
        auto& src_from = data[state_ids[i]];
        for(int j=0; j<N_STATES; j++)
        {
            auto& src_to = src_from[state_ids[j]];
            int idx = i * N_STATES + j;
            biases[idx].angle_plus = src_to["positive_bias"].get<float>();
            biases[idx].angle_minus = src_to["negative_bias"].get<float>();
        }
    }
}

void upload_exit_models(TransitionModelHost* h_exit_models)
{
    cudaMemcpyToSymbol(
        d_exit_models,
        h_exit_models,
        sizeof(TransitionModel) * N_STATES
    );
}

void upload_transition_models(TransitionModelHost* h_models)
{
    cudaMemcpyToSymbol(
        d_transition_models,
        h_models,
        sizeof(TransitionModel) * N_STATES * N_STATES
    );
}

void upload_transition_factors(TransitionFactorHost* h_factors)
{
    cudaMemcpyToSymbol(
        d_transition_factors,
        h_factors,
        sizeof(TransitionFactor) * N_STATES * N_STATES
    );
}

void upload_biases(TransitionBiasHost* h_biases)
{
    cudaMemcpyToSymbol(
        d_transition_biases,
        h_biases,
        sizeof(TransitionBias) * N_STATES * N_STATES
    );
}

float agent_kappas[9] = {4.8f, 4.27f, 4.44f, 3.62f, 3.76f, 4.11f, 3.51f, 3.33f, 2.87f};
int agent_periods[9] = {9,8,10,8,8,17,8,8,6};
float agent_amplitudes[9] = {0.5577118459338024f, 0.6163957261155478f, 0.6065887113130718f,
    0.5783402420511854f, 0.49842090781910037f, 0.6892492114529373f, 0.5892487546943024f};


__constant__ float d_agent_kappas[9];
__constant__ int d_agent_periods[9];
__constant__ float d_agent_amplitudes[9];

// CUDA kernel to initialize the position of each agent
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed, int worm_count, int agent_id) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        curand_init(seed, id, 0, &states[id]);
        if (ENABLE_RANDOM_INITIAL_POSITIONS) {
            agents[id].x = curand_uniform(&states[id]) * WIDTH;
            agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        } else {
            //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
           agents[id].x = WIDTH / 4 - INITIAL_AREA_NUMBER_OF_CELLS/2 * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
           agents[id].y = HEIGHT / 2 - INITIAL_AREA_NUMBER_OF_CELLS/2  * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
            //agents[id].x = 43.000000;
            //agents[id].y = 30.000000;
            //initialise at the center
            agents[id].x = WIDTH / 2;
            agents[id].y = HEIGHT / 2;
            //add random offset of 1mm
            agents[id].x += (curand_uniform(&states[id]) - 0.5f) * sqrt(10.0f);
            agents[id].y += (curand_uniform(&states[id]) - 0.5f) * sqrt(10.0f);


        }
        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = 0.0f;
        agents[id].angle_change = 0.0f;
        agents[id].previous_angle = agents[id].angle;
        agents[id].previous_speed = agents[id].speed;
        agents[id].is_persistent = false;
        agents[id].state_duration = 1;
        agents[id].initial_state_duration = 1;
        agents[id].previous_mag_angle_change = 0.0f;
        agents[id].p_same_sign = 0.5f;
        agents[id].phi = 0.0f;
        agents[id].agent_id = agent_id;
        agents[id].run_omega = 0.0f;
        agents[id].run_amp = 0.0f;
        agents[id].kappa =3.0f;// 2.0f + 5.0f * curand_uniform(&states[id]);
        if(agent_id>=37){
            agents[id].run_omega = 3.0f * M_PI / d_agent_periods[agent_id - 37];
            agents[id].run_amp = d_agent_amplitudes[agent_id - 37];
            agents[id].kappa = d_agent_kappas[agent_id - 37];
        }

        float generated_value = curand_uniform(&states[id]);
        agents[id].state = static_cast<int>(generated_value*(N_STATES));
        agents[id].previous_state = agents[id].state;
        //fill up dc_int with 0s
        for(int i=0; i<100; i++){
            agents[id].c[i] = 0.0f;
        }
        for(int i=0; i<N_STATES*N_STATES; i++){
            agents[id].dc_int[i] = 0.0f;
        }
        agents[id].accumulated_dc_tot = 0.0f;
        agents[id].angle_sign = 1;
        agents[id].neighbor_count = 0;
        agents[id].prev_neighbor_count = 0;
        agents[id].delta_neighbor_count = 0;
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
