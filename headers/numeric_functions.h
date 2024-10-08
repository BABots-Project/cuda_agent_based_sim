#ifndef UNTITLED_NUMERIC_FUNCTIONS_H
#define UNTITLED_NUMERIC_FUNCTIONS_H
#include <cuda_runtime.h>

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

#endif