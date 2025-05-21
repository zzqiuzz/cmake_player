// #include "matrix_add.h"

__global__ void matrix_add_kernel(float *matrix_a, float *matrix_b, float *result, int nx, int ny) {
    // 横向x, 竖向y
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = row * nx + col;

    if (row < ny && col < nx) {
        result[tid] = matrix_a[tid] + matrix_b[tid];
    }
}

void launch_matrix_add(float *matrix_a_dev, float *matrix_b_dev, float* result_dev, int nx, int ny) {

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    matrix_add_kernel<<<grid, block>>>(matrix_a_dev, matrix_b_dev, result_dev, nx, ny);
    cudaDeviceSynchronize();

}