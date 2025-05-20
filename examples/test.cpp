#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vector_add.h"
#include <cuda_runtime.h>
#include <cuda.h>
#define N 1024
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]){
    // allocate host memory
    int nByte = N * sizeof(float);
    float *out = (float*)malloc(nByte);
    float *a = (float*)malloc(nByte);
    float *b = (float*)malloc(nByte);
    // initialize host array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    // main host cpu fucntion
    vector_add_cpu(out, a, b, N);

    // verification
    for(int i = 0; i < N; i++)
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    printf("---------run add on cpu -------passed\n"); 
 
    // float *d_a, *d_b, *d_out;
    // // Allocate device memory
    // cudaMalloc((void**)&d_a, sizeof(float) * N);
    // cudaMalloc((void**)&d_b, sizeof(float) * N);
    // cudaMalloc((void**)&d_out, sizeof(float) * N);

    // cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    // // launch kernel  
    // vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    
    // // Transfer data back to host memory
    // cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // // Verification
    // for(int i = 0; i < N; i++){
    //     assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    // }
    // printf("out[0] = %f\n", out[0]);
    // printf("PASSED\n");

    // // Deallocate device memory
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
    return 0;
}