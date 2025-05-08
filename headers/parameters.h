// config.h

// Environmental variables
#define OPTIMISING true
#define N 256
#define WIDTH 60.0f
#define HEIGHT 60.0f
#define SEED 1233
// Simulation parameters
#define WORM_COUNT 1000
#define N_STEPS 1800
#define LOGGING_INTERVAL 1
#define DT 1.0f
#define DEBUG false
#define ENABLE_RANDOM_INITIAL_POSITIONS false
#define INITIAL_AREA_NUMBER_OF_CELLS 10
#define ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL true
#define MAXIMUM_AGENTS_PER_CELL 40
#define LOG_POTENTIAL false
#define LOG_GRID false
#define LOG_PHEROMONES false
#define LOG_AGENT_COUNT_GRID false
#define LOG_GENERIC_TARGET_DATA true
#define LOG_POSITIONS true
#define LOG_ANGLES true
#define LOG_VELOCITIES true
#define LOG_REPULSIVE_PHEROMONE true
#define LOG_BACTERIAL_LAWN true


// Agent parameters

//#define N_STATES 6
//std::string* state_ids = new std::string[N_STATES]{"sharp_turn", "reversal", "pause", "line", "arc", "loop"};
//std::string* state_ids = new std::string[N_STATES]{"0", "1", "2", "3", "4", "5"};
#define N_STATES 5
std::string* state_ids = new std::string[N_STATES]{"0", "1", "2", "3", "4"};

#define SENSING_RADIUS 0.01f
#define SPEED 0.1f
#define SENSING_RANGE 1
#define ODOR_THRESHOLD 1e-4f
#define POTENTIAL_THRESHOLD 1e-4f
#define ON_FOOD_SPEED_SCALE 0.09f// scale = exp(mu) => mu = log(scale)
#define ON_FOOD_SPEED_SHAPE 0.49f// shape = sigma = 0.5 these values match the tracking data with a KS test 73% of the time < 0.05 and a p-value > 0.05 33% of the time

#define OFF_FOOD_SPEED_SCALE_SLOW 0.04560973690586389f// scale = exp(mu) => mu = log(scale)
#define OFF_FOOD_SPEED_SHAPE_SLOW 0.8639747453915111f // shape = sigma
#define OFF_FOOD_SPEED_SLOW_WEIGHT 0.32525679799071244f
#define OFF_FOOD_SPEED_SCALE_FAST 0.1108085996475689f// scale = exp(mu) => mu = log(scale)
#define OFF_FOOD_SPEED_SHAPE_FAST 0.41054932407186207f // shape = sigma

#define ON_FOOD_SPEED_SCALE_SLOW 0.03649643784913813f// scale = exp(mu) => mu = log(scale)
#define ON_FOOD_SPEED_SHAPE_SLOW 0.811468808463198f // shape = sigma
#define ON_FOOD_SPEED_SLOW_WEIGHT 0.059245291380037125f
#define ON_FOOD_SPEED_SCALE_FAST 0.08978393019531566f// scale = exp(mu) => mu = log(scale)
#define ON_FOOD_SPEED_SHAPE_FAST 0.4672251563729966f // shape = sigma

#define LOOP_TIME_MU 4.38f//0.903089987f//
#define LOOP_TIME_SIGMA 0.81f
#define ARC_TIME_MU 3.89f//0.698970004//3.89f
#define ARC_TIME_SIGMA 0.81f
#define LINE_TIME_MU 3.00f//0.301029996f//3.00f
#define LINE_TIME_SIGMA 0.81f

#define PIROUETTE_TO_RUN_THRESHOLD 1e-4f
#define AUTO_TRANSITION_PROBABILITY_THRESHOLD 0.15f
#define KAPPA 7.5f
#define MAX_ALLOWED_SPEED 0.6f

// Odor parameters
#define MU_X 45.0f      // Mean x of the Gaussian
#define MU_Y 40.0f      // Mean y of the Gaussian
#define A 0.5f         // Amplitude of the Gaussian
#define SIGMA_X 10.0f   // Standard deviation in x direction
#define SIGMA_Y 10.0f   // Standard deviation in y direction
#define TARGET_AREA_SIDE_LENGTH 40
#define MAX_CONCENTRATION 0.0f
#define GAMMA 0.0001f
#define DIFFUSION_CONSTANT 0.005f //more than 0.01, it explodes
#define ATTRACTION_STRENGTH 0.0282f
#define ATTRACTION_SCALE 1.0f

// Pheromone parameters
#define ATTRACTANT_PHEROMONE_SCALE 15.0f
#define ATTRACTANT_PHEROMONE_STRENGTH 0.0f//0.000282f
#define ATTRACTANT_PHEROMONE_DECAY_RATE 0.001f
#define ATTRACTANT_PHEROMONE_SECRETION_RATE 0.00001f
#define ATTRACTANT_PHEROMONE_DIFFUSION_RATE 0.0001f
#define REPULSIVE_PHEROMONE_SCALE 1.5f
#define REPULSIVE_PHEROMONE_STRENGTH (-0.00001f)
#define REPULSIVE_PHEROMONE_DECAY_RATE 0.0005f
#define REPULSIVE_PHEROMONE_SECRETION_RATE 0.00001f
#define REPULSIVE_PHEROMONE_DIFFUSION_RATE 0.0000001f

// Bacterial lawn parameters
#define N_FOOD_SPOTS 5
float* bacterial_densities = new float[N_FOOD_SPOTS]{0.01, 0.01, 2.0, 10.0, 5.0};
#define SPOT_SIZE 10.0f
#define SPOT_DISTANCE 13.0f
#define BACTERIAL_CONSUMPTION 1e-8
#define BACTERIAL_ATTRACTION_STRENGTH 5e-5
#define BACTERIAL_ATTRACTION_SCALE 1.5f

// Noise parameters
#define SIGMA 1e-8f
#define ENVIRONMENTAL_NOISE 0.0f

// CUDA parameters
#define BLOCK_SIZE 32

__constant__ float DX = WIDTH/N;
__constant__ float DY = HEIGHT/N;