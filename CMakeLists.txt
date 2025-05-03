cmake_minimum_required(VERSION 3.0)

project(RepoForCMake)
 

add_executable(main main.cpp)

target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR/include})

#include several header file from different directories.
# target_include_directories(myapp PRIVATE
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/third_party/include
# )