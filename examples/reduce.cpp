#include <iostream>
#include "reduce.h"
#include <cuda_runtime.h>
#include <cuda.h>

#define ThreadPerBlock 256

int main(int argc, char *argv[]){
    // const int N = 32 * 1024 * 1024; 
    const int N = 257;
    int blockNum =(N + ThreadPerBlock - 1) / ThreadPerBlock;

    /*-----------------------------compute on cpu------------------------*/
    float *host_input_data = (float*)malloc(N * sizeof(float));
    // initialize host input data
    for(int i = 0; i < N; i++)
        host_input_data[i] = static_cast<float>(int(1));
    float *out = (float*)malloc(blockNum * sizeof(float));
    for(int i = 0; i < blockNum; i++){
        float cur_per_block_sum = 0.f;
        for(int j = 0; j < ThreadPerBlock; j++){
            if(i * ThreadPerBlock + j < N)
                cur_per_block_sum += host_input_data[i * ThreadPerBlock + j];
        }
        out[i] = cur_per_block_sum;
    }

    /*-----------------------------compute on gpu------------------------*/
    float *device_input_data = nullptr; 
    cudaMalloc((void**)&device_input_data, N * sizeof(float));
    cudaMemcpy(device_input_data, host_input_data, N * sizeof(float), cudaMemcpyHostToDevice);
    float *device_out = nullptr;
    cudaMalloc((void**)&device_out, blockNum * sizeof(float));

    reduce_baseline(device_input_data, device_out, ThreadPerBlock, N);
    float* out_data_d2h = (float*)malloc(blockNum * sizeof(float));
    cudaMemcpy(out_data_d2h, device_out, blockNum * sizeof(float), cudaMemcpyDeviceToHost);
    
    if(check(out, out_data_d2h, blockNum)){
        std::cout << "GPU result is correct!" << std::endl;
    }else{
        std::cout << "GPU result is wrong!" << std::endl;
        for(int i=0;i<blockNum;i++){
            printf("%lf ",out_data_d2h[i]);
        }
        printf("\n");
    }

    free(host_input_data);
    cudaFree(device_input_data);
    free(out);
    free(out_data_d2h);
    cudaFree(device_out);   




//https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/reduce/reduce_v0_baseline.cu

//https://www.bilibili.com/video/BV1HvBSY2EJW?spm_id_from=333.788.videopod.episodes&vd_source=64d595dda97cc94139448e5bedc1774e&p=2

 

}