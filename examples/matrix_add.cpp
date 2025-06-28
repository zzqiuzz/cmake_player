#include "matrix_helper.h"
#include <cstring>

int main(int argc, char *argv[]) { 

    int nx = 1024, ny = 1024;
    int total_size = nx * ny;
    float *matrix_a_host = (float*)malloc(total_size * sizeof(float));
    float *matrix_b_host = (float*)malloc(total_size * sizeof(float));
    float *result_host = (float*)malloc(total_size * sizeof(float));
    init_matrix(matrix_a_host, nx, ny);
    init_matrix(matrix_b_host, nx, ny);
    memset(result_host, 0, total_size * sizeof(float));

 
    //cudamalloc
    float *matrix_a_dev = nullptr;
    float *matrix_b_dev = nullptr;
    float *result_dev = nullptr;
    cudaMalloc((void**)&matrix_a_dev, total_size * sizeof(float));
    cudaMalloc((void**)&matrix_b_dev, total_size * sizeof(float));
    cudaMalloc((void**)&result_dev, total_size * sizeof(float));
    cudaMemcpy(matrix_a_dev, matrix_a_host, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_dev, matrix_b_host, total_size * sizeof(float), cudaMemcpyHostToDevice);



    //kernel launch
    launch_matrix_add(matrix_a_dev, matrix_b_dev, result_dev, nx, ny); 
    cudaMemcpy(result_host, result_dev, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(result_host, nx, ny);

    cudaFree(matrix_a_dev);
    cudaFree(matrix_b_dev); 
    cudaFree(result_dev);
    free(matrix_a_host);
    free(matrix_b_host);
    free(result_host);
    cudaDeviceReset();    

    return 0;
}