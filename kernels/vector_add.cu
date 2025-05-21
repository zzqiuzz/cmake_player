// vector_add.cu
#include <cuda_runtime.h>
#include <iostream>

// CUDA核函数：两个向量相加
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
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
        a[i] = i;
        b[i] = i * 2;
    }

    // 分配设备内存
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数
    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);

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