#pragma once


#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
using std::cout;
using std::endl;

void init_matrix(float *matrix, int nx, int ny){
    for(int row = 0; row < nx; row++){
        for(int col = 0; col < ny; col++){ 
            matrix[row * ny + col] = 1.f;
        }
    }
    std::cout << "Matrix initialized" << std::endl;
}

void print_matrix(float *matrix, int nx, int ny){
    for(int row = 0; row < nx; row++){
        cout << "-------------------------" << endl;
        for(int col = 0; col < ny; col++){ 
            cout << matrix[row * ny + col] << " ";
        }
        cout << endl << "-------------------------" << endl;
    } 
}


void launch_matrix_add(float*, float*, float*, int, int); 