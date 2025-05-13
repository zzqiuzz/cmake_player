#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
void vector_add_cpu(float*, const float*, const float*, int); 
void launch_add(float*, const float*, const float*, int); 