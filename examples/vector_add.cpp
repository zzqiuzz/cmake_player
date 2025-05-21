// main.cpp
#include <iostream>

// 声明外部函数（来自vector_add.cu）
// extern "C" void runVectorAdd(int n);
#include "vector_add.h"

int main() {
    int n = 1 << 10; // 向量长度
    std::cout << "Running vector addition on GPU..." << std::endl;
    runVectorAdd(n);
    return 0;
}