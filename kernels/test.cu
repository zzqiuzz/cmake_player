#include <iostream>
#include <cuda_runtime.h>

#define N 257
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 2

// CUDA Reduce Kernel
__global__ void reduceSumKernel(const float* input, float* output) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 将输入载入共享内存，注意边界检查
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // 规约求和：从 thread 1 到 0
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // block 0 和 block 1 的结果写入 output[0] 和 output[1]
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    float h_input[N];
    float h_output[NUM_BLOCKS];

    // 初始化输入数组
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // 可以改为任意值测试
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, NUM_BLOCKS * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    reduceSumKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output);

    // 把结果拷贝回主机
    cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = h_output[0] + h_output[1];

    std::cout << "Final Sum: " << finalSum << std::endl;

    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}