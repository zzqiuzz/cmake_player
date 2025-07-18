#pragma once


#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream> 
using std::cout;
using std::endl;
#define TileWidth 16

enum class matmulCalType : uint8_t{
    naive = 0,
    tiled = 1,
};

void init_matrix(float *matrix, int nx, int ny){
    for(int row = 0; row < nx; row++){
        for(int col = 0; col < ny; col++){ 
            matrix[row * ny + col] = 1.f + row % 32;
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

void matmul_cpu(const float *a, const float *b, float *result, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++) {
                result[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

void matmul_trans_cpu(const float *a, float *trans_a, int  M, int N) {
    for(int n = 0; n < N; n++){
        for(int m = 0; m < M; m++){
            trans_a[n * M + m] = a[m * N + n];
        }
    }
     
}

void check(float *a, float *b, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i * n + j] != b[i * n + j]) {
                cout << "Mismatch at (" << i << ", " << j << "): "
                     << a[i * n + j] << " != " << b[i * n + j] << endl;
                return;
            }
        }
    }
    cout << "Matrices match!" << endl;
}


void launch_matrix_add(float*, float*, float*, int, int); 
void launch_matmul_naive(const float *A, const float *B, float *result, int M, int N, int K); 
void launch_matmul_tiled(const float *A, const float *B, float *result, int M, int N, int K); 
void launch_matrix_trans_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N); 
void launch_matrix_trans_read_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N); 
void launch_matrix_trans_read_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N); 