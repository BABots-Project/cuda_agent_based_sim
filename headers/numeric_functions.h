#ifndef UNTITLED_NUMERIC_FUNCTIONS_H
#define UNTITLED_NUMERIC_FUNCTIONS_H
#include <cuda_runtime.h>




//function to sample from a wrapped cauchy distribution
__device__ float wrapped_cauchy(float mu, float sigma, curandState* state){
    float u = curand_uniform(state);
    float theta = mu + sigma * tan(M_PI * (u - 0.5f));

    //wrap the angle between -pi and pi
    if(theta > M_PI){
        theta -= 2 * M_PI;
    }
    if(theta <= -M_PI){
        theta += 2 * M_PI;
    }
    return theta;
}


// Function to compute the gradient in the X direction (partial derivative)
__device__ float gradientX(float* grid, int i, int j) {
    // Periodic boundary conditions
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += N;
    int rightIndex = i + 1;
    if (rightIndex >= N) rightIndex -= N;
    float left = grid[leftIndex * N + j];
    float right = grid[rightIndex * N + j];

    return (right - left) / (2.0f * DX);  // Central difference
}

// Function to compute the gradient in the Y direction (partial derivative)
__device__ float gradientY(float* grid, int i, int j) {
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += N;
    int upIndex = j + 1;
    if (upIndex >= N) upIndex -= N;
    float down = grid[i * N + downIndex];
    float up = grid[i * N + upIndex];

    return (up - down) / (2.0f * DX);  // Central difference
}

// Function to compute the Laplacian (second derivative)
__device__ float laplacian(float* grid, int i, int j) {
    float center = grid[i * N + j];
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += N;
    int rightIndex = i + 1;
    if (rightIndex >= N) rightIndex -= N;
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += N;
    int upIndex = j + 1;
    if (upIndex >= N) upIndex -= N;
    float left = grid[leftIndex * N + j];
    float right = grid[rightIndex * N + j];
    float down = grid[i * N + downIndex];
    float up = grid[i * N + upIndex];

    float laplacian = (left + right + up + down - 4.0f * center) / (DX * DX);
    if (isnan(laplacian) || isinf(laplacian)) {
        printf("Invalid laplacian %f at (%d, %d)\n", laplacian, i, j);
        //printf("Center %f\n", center);
        //printf("Left %f\n", left);
        //printf("Right %f\n", right);
        //printf("Down %f\n", down);
        //printf("Up %f\n", up);
    }
    return laplacian;
}

// function to compute the 4th order laplacian
__device__ float fourth_order_laplacian(float* input, int i, int j){
    int im2 = (i - 2 + N) % N;
    int im1 = (i - 1 + N) % N;
    int ip1 = (i + 1) % N;
    int ip2 = (i + 2) % N;

    int jm2 = (j - 2 + N) % N;
    int jm1 = (j - 1 + N) % N;
    int jp1 = (j + 1) % N;
    int jp2 = (j + 2) % N;

    float laplacianX = (-input[im2 * N + j] + 16 * input[im1 * N + j] - 30 * input[i * N + j]
                        + 16 * input[ip1 * N + j] - input[ip2 * N + j]) / (12 * DX * DX);

    float laplacianY = (-input[i * N + jm2] + 16 * input[i * N + jm1] - 30 * input[i * N + j]
                        + 16 * input[i * N + jp1] - input[i * N + jp2]) /  (12 * DX * DX);

    return laplacianX + laplacianY;
}

#endif