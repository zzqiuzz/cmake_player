#include <iostream>
#include <matrix_add.h>
using std::cout;
using std::endl;

#define M 256
#define N 256
#define K 256

int main(int argc, char *argv[]){

    float a[M][K];
    float b[K][N]; // notice row major or col major
    float result[M][N] = {0.f};
    init_matrix((float*)a, M, K);
    init_matrix((float*)b, N, K);
    matmul_cpu((float*)a, (float*)b, (float*)result, M, N, K);

    float *d_a, *d_b, *d_result;
    float *result_d2h = (float*)malloc(M * N * sizeof(float));
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, N * K * sizeof(float));
    cudaMalloc((void**)&d_result, M * N * sizeof(float));


    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice); 

    // launch kernel
    if(0)
        launch_matmul_naive(d_a, d_b, d_result, M, N, K);
    else
        launch_matmul_tiled(d_a, d_b, d_result, M, N, K);

    cudaMemcpy(result_d2h, d_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    check((float*)result, (float*)result_d2h, M, N, K);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result); 
    free(result_d2h);
    return 0;
}