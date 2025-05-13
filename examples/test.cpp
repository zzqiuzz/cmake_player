#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vector_add.h"
#define N 1024
#define MAX_ERR 1e-6


int main(int argc, char *argv[]){
    // allocate host memory
    float *out = (float*)malloc(N * sizeof(float));
    float *in_0 = (float*)malloc(N * sizeof(float));
    float *in_1 = (float*)malloc(N * sizeof(float));
    // initialize host array
    for(int i = 0; i < N; i++){
        in_0[i] = 1.0f;
        in_1[i] = 2.0f;
    }
    // main host cpu fucntion
    vector_add_cpu(out, in_0, in_1, N);

    // verification
    for(int i = 0; i < N; i++)
        assert(fabs(out[i] - in_0[i] - in_1[i]) < MAX_ERR);
    printf("----------------passed\n"); 

    // cuda
    cuda


    free(out);
    free(in_0);
    free(in_1);
    return 0;
}