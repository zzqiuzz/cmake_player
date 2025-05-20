cmake_minimum_required(VERSION 3.0)

project(TEST)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

# set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set include path
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/utils
                    /usr/local/cuda/include)       

# get source cpp                     
set(TESTSOURCES examples/test.cpp kernels/vector_add.cpp)
set(MAINSOURCES examples/main.cpp)

add_executable(test ${TESTSOURCES})
add_executable(main ${MAINSOURCES})
 

#include several header file from different directories.
# target_include_directories(myapp PRIVATE
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/third_party/include
# )