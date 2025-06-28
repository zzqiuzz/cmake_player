#define TileWidth 16
#include <iostream>
using std::cout;
using std::endl;

__global__ void matrix_trans_naive(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < M)
        matrix_trans_dev[row * M + col] = matrix_dev[col * N + row]; // 写入时连续的 但是读取不是连续的，不连续则会导致严重的非合并访问
}
 

void launch_matrix_trans(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((M + TileWidth - 1) / TileWidth, (N + TileWidth - 1) / TileWidth);
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matrix_trans_naive<<<grid, block>>>(matrix_dev, matrix_trans_dev, M, N);
    // transpose_shared<<<grid, block>>>(matrix_dev, matrix_trans_dev, M, N);
    // shared memory
    // deliminate bank conflicts
    cudaDeviceSynchronize();
}