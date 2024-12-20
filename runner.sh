#!/bin/bash

# Compile the main.cu file to an executable called "main"
nvcc main.cu -o main -lm --expt-relaxed-constexpr

# Define the arrays for W, br, ba, and range for n_exp
W_vals=(1 10 100 1000 10000 100000)
br_vals=(0 1e-5 1e-3 1e-1 10)
ba_vals=(0 1e-5 1e-3 1e-1 10)
n_exp_start=0
n_exp_end=20

#W_vals=(1)
#br_vals=(1e-4 1e-6 1e-8 0)
#ba_vals=(1e-4 1e-6 1e-8 0)

# Source and destination directories
#src_logs="/home/nema/CLionProjects/untitled/logs/"
src_agents_log="/home/nema/CLionProjects/untitled/agents_all_data.json"


echo "Add odor? (y/n)"
read add_odor
#add_odor="y"
# Determine which checkpoint file to use based on input
if [ "$add_odor" == "y" ]; then
    checkpoint_file="checkpoint_odor.txt"
    base_dst="/home/nema/cuda_worm_sim/data/odor_and_pheromones"
    add_odor_bool=1
else
    checkpoint_file="checkpoint_no_odor.txt"
    base_dst="/home/nema/cuda_worm_sim/data/no_odor_and_pheromones"
    add_odor_bool=0
fi

if [ -f $checkpoint_file ]; then
    #echo "Checkpoint file exists. Do you want to delete it and restart from scratch? (y/n)"
    #read answer
    answer="y"
    if [ "$answer" = "y" ]; then
        rm $checkpoint_file
        echo "Checkpoint deleted."
    else
        echo "Continuing from the last checkpoint."
    fi
fi

# Initialize the checkpoint file if it doesn't exist
if [ ! -f $checkpoint_file ]; then
    echo "0 0 0 0" > $checkpoint_file
fi

# Read the last completed step from the checkpoint file
read last_W_idx last_br_idx last_ba_idx last_n_exp < $checkpoint_file

# Loop through all combinations of W, br, ba, and n_exp
for ((W_idx=last_W_idx; W_idx<${#W_vals[@]}; W_idx++)); do
    W=${W_vals[$W_idx]}
    for ((br_idx=(W_idx==last_W_idx ? last_br_idx : 0); br_idx<${#br_vals[@]}; br_idx++)); do
        br=${br_vals[$br_idx]}
        for ((ba_idx=(br_idx==last_br_idx && W_idx==last_W_idx ? last_ba_idx : 0); ba_idx<${#ba_vals[@]}; ba_idx++)); do
            ba=${ba_vals[$ba_idx]}
            for ((n_exp=(ba_idx==last_ba_idx && br_idx==last_br_idx && W_idx==last_W_idx ? last_n_exp : $n_exp_start); n_exp<=$n_exp_end; n_exp++)); do

                #empty the logs folder
                #rm -r "/home/nema/CLionProjects/untitled/logs/agent_count/*"
                #rm -r "/home/nema/CLionProjects/untitled/logs/attractive_pheromone/*"
                #rm -r "/home/nema/CLionProjects/untitled/logs/repulsive_pheromone/*"
                #rm -r "/home/nema/CLionProjects/untitled/logs/chemical_concentration/*"
                #rm -r "/home/nema/CLionProjects/untitled/logs/potential/*"

                # Create destination directories
                #dst_logs="$base_dst/W_$W/beta_a_$ba/beta_r_$br/exp_n_$n_exp/logs/"
                dst_agents_log="$base_dst/W_$W/beta_a_$ba/beta_r_$br/exp_n_$n_exp/agents_all_data.json"


                # Make sure destination directories exist
                #mkdir -p "$dst_logs"
                mkdir -p "$base_dst/W_$W/beta_a_$ba/beta_r_$br/exp_n_$n_exp"
                # Run the main executable with the current parameters
                echo "Running with W=$W, br=$br, ba=$ba, n_exp=$n_exp, odor=$add_odor_bool"
                ./main $n_exp $W $ba $br $add_odor_bool

                # Copy logs folder and individual files
                echo "Copying logs and data to $dst_logs and other destinations"
                #cp -r "$src_logs" "$dst_logs"
                cp "$src_agents_log" "$dst_agents_log"

                # Save the current progress to the checkpoint file
                echo "$W_idx $br_idx $ba_idx $n_exp" > $checkpoint_file
            done
        done
    done
done