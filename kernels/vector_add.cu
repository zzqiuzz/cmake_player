#include <stdio.h>
__global__ void vector_add(float *out, const float *in_0, const float *in_1, int n){
    for(int i = 0; i < n; i++)
        out[i] = in_0[i] + in_1[i];
}

void launch_add(float *out, const float *in_0, const float *in_1, int n){
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);
    vector_add<<<grid, block>>>(out, in_0, in_1, n);
}