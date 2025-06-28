#include "matrix_helper.h"
#include <cstring>
#define M 128
#define N 40

int main(int argc, char *argv[]) { 
 
    int total_size = M * N;
    float matrix_host[M][N]  ;
    float matrix_trans_cpu[N][M]; 
    float matrix_trans_host[N][M]; 
    init_matrix((float*)matrix_host, M, N);  
    // print_matrix(matrix_host, M, N);
    matmul_trans_cpu((float*)matrix_host, (float*)matrix_trans_cpu, M, N);
    // print_matrix(matrix_trans_cpu, N, M);
 
    //cudamalloc
    float *matrix_dev = nullptr;
    float *matrix_trans_dev = nullptr; 
    cudaMalloc((void**)&matrix_dev, total_size * sizeof(float));
    cudaMalloc((void**)&matrix_trans_dev, total_size * sizeof(float)); 
    cudaMemcpy(matrix_dev, matrix_host, total_size * sizeof(float), cudaMemcpyHostToDevice); 



    //kernel launch
    launch_matrix_trans(matrix_dev, matrix_trans_dev, M, N); 
    cudaMemcpy(matrix_trans_host, matrix_trans_dev, total_size * sizeof(float), cudaMemcpyDeviceToHost); 
    check((float*)matrix_trans_cpu, (float*)matrix_trans_host, N, M);

    cudaFree(matrix_dev);
    cudaFree(matrix_trans_dev);   
    cudaDeviceReset();    

    return 0;
}