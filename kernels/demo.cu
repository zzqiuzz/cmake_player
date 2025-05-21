#include <iostream>
#include "test.h"
using namespace std;


__global__ void hello_gpu() 
{
    // 在GPU上打印Hello World
    printf("Hello World from GPU!\n");
}

void fun()
{

    hello_gpu<<<4, 4>>>(); // 启动内核，4个块，每个块有4个线程，执行16次hello_gpu()函数调用。

    cudaDeviceSynchronize(); // 等待GPU完成所有工作
    printf("Hello World from CPU!\n");
} 