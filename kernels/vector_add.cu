// vector_add.cu
#include <cuda_runtime.h>
#include <iostream>
/*
if n = 100000; block_size = 256; gridDim.x = n / block_size / 5 (e.g)
n的个数特别大，以至于gridDim.x * blocksize覆盖不了整个n，因此每个线程就需要多处理元素了，这里用+=stride来处理
__global__ void vectorAdd(int* a, int* b, int* c, int n) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
            index                                                          stride = blockDim.x * gridDim.x
    [---------*------,--------------,--------------,----------------,=========*===============================] totally N elements
    <---blockDim.x--->
    <------------------------------gridDim.x------------------------>
 */


// CUDA核函数：两个向量相加
__global__ void vectorAdd(int* a, int* b, int* c, int n) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid]; 
    }
}

// 外部接口函数，供main.cpp调用
void runVectorAdd(int n) {
    int *a, *b, *c;              // 主机指针
    int *d_a, *d_b, *d_c;        // 设备指针

    // 分配主机内存
    a = new int[n];
    b = new int[n];
    c = new int[n];

    // 初始化输入数据
    for (int i = 0; i < n; ++i) { 
        a[i] = 0;
        b[i] = i; 
    }

    // 分配设备内存
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);
 
 
    dim3 block(256); // 三维数组， block.x = 256, block.y = 1, block.z = 1
    dim3 grid((n + block.x - 1) / block.x);// 三维数组， grid.x = (n + block.x - 1) / block.x, grid.y = 1, grid.z = 1
    // 计算网格和块的大小
    // 这里的 grid.x 是计算出的网格大小， block.x 是块大小 
    printf("Grid size: %d, Block size: %d\n", grid.x, block.x);

    // 启动核函数
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n); 

    // 将结果从设备复制回主机
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // 释放资源
    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}