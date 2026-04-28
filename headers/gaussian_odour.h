//
// Created by nema on 28/10/24.
//

#ifndef UNTITLED_GAUSSIAN_ODOUR_H
#define UNTITLED_GAUSSIAN_ODOUR_H
#include <cuda_runtime.h>


__device__ float diffusionProfile(float x, float y, float t, float D=2.52e-2f){
    if (t <= 0) {
        return 0.0f; // No diffusion at time t=0
    }

    return MAX_CONCENTRATION * expf(-((x-odor_x0)*(x-odor_x0)+(y-odor_y0)*(y-odor_y0))/(4*DIFFUSION_CONSTANT*(t+ODOR_T0)));
}

__device__ float gaussianDensity(float x, float y, float mu_x, float mu_y, float a, float sigma_x, float sigma_y) {
    // Compute the 2D Gaussian density for a single Gaussian centered at (mu_x, mu_y)
    float dx = x - mu_x;
    float dy = y - mu_y;

    // Calculate Gaussian exponent
    float exponent = -(dx * dx / (2 * sigma_x * sigma_x) + dy * dy / (2 * sigma_y * sigma_y));

    // Return the density value
    return a * expf(exponent) / (2.0f * M_PI * sigma_x * sigma_y);
}

__global__ void computeDensityKernel(float *output, float x, float y, float mu_x, float mu_y, float a, float sigma_x, float sigma_y, float Lx, float Ly) {
    // This kernel calculates the density at point (x, y) with zero boundaries outside the domain

    // Thread index (use single-thread kernel for a single density computation)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;  // Ensure only one thread executes

    // Check if the evaluation point (x, y) is within the domain [0, Lx] x [0, Ly]
    if (x < 0.0f || x > Lx || y < 0.0f || y > Ly) {
        output[0] = 0.0f;  // Zero density outside the domain
        return;
    }

    // Compute the density using a single Gaussian centered at (mu_x, mu_y)
    float density = gaussianDensity(x, y, mu_x, mu_y, a, sigma_x, sigma_y);

    // Check if the Gaussian mean (mu_x, mu_y) is outside the domain boundaries
    if (mu_x < 0.0f || mu_x > Lx || mu_y < 0.0f || mu_y > Ly) {
        density = 0.0f;  // Zero density if Gaussian is centered outside the domain
    }

    // Store the computed density in the output
    output[0] = density;
}

// device function to call the CUDA kernel
__device__ float computeDensityAtPoint(float x, float y, float t) {
    // Check if the point (x, y) is within the boundaries
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
        return 0.0f;  // Return 0 for points outside the boundaries
    }

    // Compute the Gaussian density value at (x, y)
    float dx = x - MU_X;
    float dy = y - MU_Y;
    float a_t = A * expf(-GAMMA * t);
    float sigma_x_t = SIGMA_X + 2 * DIFFUSION_CONSTANT * t;
    float sigma_y_t = SIGMA_Y + 2 * DIFFUSION_CONSTANT * t;
    float density = MAX_CONCENTRATION * a_t * expf(-0.5f * ((dx * dx) / (sigma_x_t * sigma_x_t) + (dy * dy) / (sigma_y_t * sigma_y_t)));
    //fix to a time-static value
    //density = MAX_CONCENTRATION * expf(-0.5f * ((dx * dx) / (SIGMA_X * SIGMA_X) + (dy * dy) / (SIGMA_Y * SIGMA_Y)));
    return density;
}
#endif //UNTITLED_GAUSSIAN_ODOUR_H
