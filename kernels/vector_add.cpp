#include "vector_add.h"
void vector_add_cpu(float *out, const float *in_0, const float *in_1, int size){
    for(int i = 0; i < size; i++)
        out[i] = in_0[i] + in_1[i];
}