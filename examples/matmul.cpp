#include <iostream>
#include <matrix_helper.h>
using std::cout;
using std::endl;

#define M 257
#define N 511
#define K 63

int main(int argc, char *argv[]){

    matmulCalType cal_type = matmulCalType::tiled;
    float a[M][K];
    float b[N][K]; // notice row major or col major
    float result[M][N] = {0.f};
    init_matrix((float*)a, M, K);
    init_matrix((float*)b, N, K);
    matmul_cpu((float*)a, (float*)b, (float*)result, M, N, K);

    float *d_a, *d_b, *d_result;
    // float *result_d2h = (float*)malloc(M * N * sizeof(float));
    float result_d2h[M][N] = {0.f};
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, N * K * sizeof(float));
    cudaMalloc((void**)&d_result, M * N * sizeof(float));


    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice); 

    // launch kernel
    if(cal_type == matmulCalType::naive)
        launch_matmul_naive(d_a, d_b, d_result, M, N, K);
    else if((cal_type == matmulCalType::tiled))
        launch_matmul_tiled(d_a, d_b, d_result, M, N, K);
    else
        throw("Not implemented.");

    cudaMemcpy(result_d2h, d_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    check((float*)result, (float*)result_d2h, M, N);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result); 
    // free(result_d2h);
    return 0;
}