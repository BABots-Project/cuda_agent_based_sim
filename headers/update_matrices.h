//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_UPDATE_MATRICES_H
#define UNTITLED_UPDATE_MATRICES_H
#include <cuda_runtime.h>
#include "numeric_functions.h"


//CUDA kernel to update the repulsive pheromone grid and the bacterial lawn grid
__global__ void updateRepulsivePheromoneAndBacterialLawnGrids(float* repulsive_pheromone, float* bacterial_lawn, int* agent_count_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N && i>=0 && j>=0) {
        float laplacian = fourth_order_laplacian(repulsive_pheromone, i, j);
        repulsive_pheromone[i*N+j] += DT*(REPULSIVE_PHEROMONE_DIFFUSION_RATE * laplacian + REPULSIVE_PHEROMONE_SECRETION_RATE*agent_count_grid[i*N+j] - REPULSIVE_PHEROMONE_DECAY_RATE*repulsive_pheromone[i*N+j]);
        if ( repulsive_pheromone[i*N+j]<0)  repulsive_pheromone[i*N+j]=0;
        bacterial_lawn[i*N+j] += DT*(-BACTERIAL_CONSUMPTION * agent_count_grid[i*N+j]);
        if (bacterial_lawn[i*N+j]<0) bacterial_lawn[i*N+j]=0;
    }
}


//CUDA kernel to update all the grids (except the potential and the agent count grid)
__global__ void updateGrids(float* grid, float* attractive_pheromone, float* repulsive_pheromone, int* agent_count_grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N && i>=0 && j>=0) {
        //if in the target area, set the concentration to the maximum value
        if(i>=3*N/4-TARGET_AREA_SIDE_LENGTH/2 && i<3*N/4+TARGET_AREA_SIDE_LENGTH/2 && j>=N/2-TARGET_AREA_SIDE_LENGTH/2 && j<N/2+TARGET_AREA_SIDE_LENGTH/2){
            grid[i * N + j] = MAX_CONCENTRATION;
        }
        else {
            float laplacian_value = fourth_order_laplacian(grid, i, j); //laplacian(grid, i, j);

            float new_concentration =
                    grid[i * N + j] + DT * (DIFFUSION_CONSTANT * laplacian_value - GAMMA * grid[i * N + j]);
            if (new_concentration < 0) new_concentration = 0.0f;
            if (new_concentration > MAX_CONCENTRATION) new_concentration = MAX_CONCENTRATION;
            //check if the grid is a valid float number
            if (isnan(new_concentration) || isinf(new_concentration)) {
                printf("Invalid concentration %f at (%d, %d)\n", new_concentration, i, j);
                printf("Laplacian value %f\n", laplacian_value);
                printf("Old concentration %f\n", grid[i * N + j]);

            }

            grid[i * N + j] = new_concentration;
        }
        //update attractive pheromone
        //laplacian_value = fourth_order_laplacian(attractive_pheromone, i, j);
        float new_attractive_pheromone, laplacian_attractive_pheromone = fourth_order_laplacian(attractive_pheromone, i, j);
        float new_repulsive_pheromone, laplacian_repulsive_pheromone = fourth_order_laplacian(repulsive_pheromone, i, j);
        if(agent_count_grid[i * N + j] == 0){
            new_attractive_pheromone =  attractive_pheromone[i * N + j] + DT * (ATTRACTANT_PHEROMONE_DIFFUSION_RATE * laplacian_attractive_pheromone - ATTRACTANT_PHEROMONE_DECAY_RATE * attractive_pheromone[i * N + j]);
            new_repulsive_pheromone = repulsive_pheromone[i * N + j] + DT * (REPULSIVE_PHEROMONE_DIFFUSION_RATE * laplacian_repulsive_pheromone - REPULSIVE_PHEROMONE_DECAY_RATE * repulsive_pheromone[i * N + j]);
        }
        else {
            new_attractive_pheromone = attractive_pheromone[i * N + j] + DT * (ATTRACTANT_PHEROMONE_DIFFUSION_RATE * laplacian_attractive_pheromone - ATTRACTANT_PHEROMONE_DECAY_RATE * attractive_pheromone[i * N + j] + ATTRACTANT_PHEROMONE_SECRETION_RATE * agent_count_grid[i * N + j] / (DX * DX));
            new_repulsive_pheromone = repulsive_pheromone[i * N + j] + DT * (REPULSIVE_PHEROMONE_DIFFUSION_RATE * laplacian_repulsive_pheromone - REPULSIVE_PHEROMONE_DECAY_RATE * repulsive_pheromone[i * N + j] + REPULSIVE_PHEROMONE_SECRETION_RATE * agent_count_grid[i * N + j] / (DX * DX));
        }
        if (new_attractive_pheromone < 0) new_attractive_pheromone = 0.0f;
        if (new_attractive_pheromone > MAX_CONCENTRATION) new_attractive_pheromone = MAX_CONCENTRATION;
        attractive_pheromone[i * N + j] = new_attractive_pheromone;

        if(isnan(new_attractive_pheromone) || isinf(new_attractive_pheromone)){
            printf("Invalid attractive pheromone %f at (%d, %d)\n", new_attractive_pheromone, i, j);
            printf("Laplacian value %f\n", laplacian_attractive_pheromone);
            printf("Old attractive pheromone %f\n", attractive_pheromone[i * N + j]);
        }

        //update repulsive pheromone
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * N + j]);


        repulsive_pheromone[i * N + j] = new_repulsive_pheromone;

        if(isnan(new_repulsive_pheromone) || isinf(new_repulsive_pheromone)){
            printf("Invalid repulsive pheromone %f at (%d, %d)\n", new_repulsive_pheromone, i, j);
            printf("Laplacian value %f\n", laplacian_repulsive_pheromone);
            printf("Old repulsive pheromone %f\n", repulsive_pheromone[i * N + j]);
        }
    }
}

//CUDA kernel to update the grid of the chemical concentration using a reaction-diffusion equation
__global__ void updateGrid(float* grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {

        float laplacian_value = laplacian(grid, i, j);

        float new_concentration = grid[i * N + j] + DT * (DIFFUSION_CONSTANT * laplacian_value - GAMMA * grid[i * N + j]);
        if (new_concentration < 0) new_concentration = 0.0f;
        if (new_concentration > MAX_CONCENTRATION) new_concentration = MAX_CONCENTRATION;
        //check if the grid is a valid float number
        if (isnan(new_concentration) || isinf(new_concentration)) {
            printf("Invalid concentration %f at (%d, %d)\n", new_concentration, i, j);
            printf("Laplacian value %f\n", laplacian_value);
            printf("Old concentration %f\n", grid[i * N + j]);

        }

        grid[i * N + j] = new_concentration;
    }
}

//CUDA kernel to update the potential matrix
__global__ void updatePotential(float* potential, float* grid, float* attractive_pheromone, float* repulsive_pheromone, float attractive_pheromone_strength, float repulsive_pheromone_strength, float odor_strength, curandState* states, float environmental_noise) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        float potential_odor=0.0f, potential_attractive_pheromone=0.0f, potential_repulsive_pheromone = 0.0f;
        potential_odor = odor_strength * log10(ATTRACTION_SCALE + grid[i * N + j]);// / (ATTRACTION_SCALE + grid[i * N + j]);

        //
        potential_attractive_pheromone = attractive_pheromone_strength * log10(ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);// / (ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);

        potential_repulsive_pheromone = repulsive_pheromone_strength * log10(REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);// / (REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);


        float noise = curand_normal(&states[i * N + j]) * environmental_noise;
        if(noise>environmental_noise) noise = environmental_noise;
        if(noise<-environmental_noise) noise = -environmental_noise;
        potential[i * N + j] = potential_odor + potential_attractive_pheromone + potential_repulsive_pheromone + noise;

    }
}
#endif //UNTITLED_UPDATE_MATRICES_H
