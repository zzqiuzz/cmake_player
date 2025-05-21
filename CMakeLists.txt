cmake_minimum_required(VERSION 3.0) 
project(CUDA_Project LANGUAGES CXX CUDA)  # 新增CUDA语言支持 

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

# set include path
include_directories(${PROJECT_SOURCE_DIR}/include
                    /usr/local/cuda/include)
# 输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# 添加主程序 

add_executable(demo examples/demo.cpp kernels/demo.cu) 
target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

add_executable(vector_add examples/vector_add.cpp kernels/vector_add.cu) 
target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

add_executable(matrix_add examples/matrix_add.cpp kernels/matrix_add.cu) 
target_compile_options(matrix_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

 

# Terminal -> run task -> cmake --build