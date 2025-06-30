#define TileWidth 32
#include <iostream>
using std::cout;
using std::endl;

__global__ void matrix_trans_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < M)
        matrix_trans_dev[row * M + col] = matrix_dev[col * N + row]; // 写入是连续的 读取不是连续的，不连续则会导致严重的非合并访问
}
 

void launch_matrix_trans_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((M + TileWidth - 1) / TileWidth, (N + TileWidth - 1) / TileWidth); // 这里是对输出矩阵进行block划分
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matrix_trans_write_coalesced<<<grid, block>>>(matrix_dev, matrix_trans_dev, M, N); 
    cudaDeviceSynchronize();
}
/*
                N  col bx
        -------------------
    by  |                 |
        |                 |
        |                 |
    row |           *     |  row = blockIdx.y * blockDim.y + threadIdx.y;
        |                 |  col = blockIdx.x * blockDim.x + threadIdx.x;
     M  |                 |
        |                 |
        |                 |
        |                 |
        |                 |
        |                 | 
        -------------------
*/

__global__ void matrix_trans_read_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N)
        matrix_trans_dev[col * M + row] = matrix_dev[row * N + col]; // 读取是连续的 写入不连续
}
 

void launch_matrix_trans_read_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((N + TileWidth - 1) / TileWidth, (M + TileWidth - 1) / TileWidth); // 这里是对输入矩阵进行block划分
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matrix_trans_read_coalesced<<<grid, block>>>(matrix_dev, matrix_trans_dev, M, N); 
    cudaDeviceSynchronize();
}

__global__ void matrix_trans_read_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){ 
    __shared__ float buffer[TileWidth][TileWidth + 1]; // avoid bank conficts
    // load data from global memory to share memory
    int matrix_row = blockIdx.y * blockDim.y + threadIdx.y;
    int matrix_col = blockIdx.x * blockDim.x + threadIdx.x;
    if(matrix_row < M && matrix_col < N)
        buffer[threadIdx.y][threadIdx.x] = matrix_dev[matrix_row * N + matrix_col]; // 读取连续
    __syncthreads();

    int matrix_trans_row = blockIdx.x * blockDim.x + threadIdx.y;
    int matrix_trans_col = blockIdx.y * blockDim.y + threadIdx.x;
    if(matrix_trans_row < N && matrix_trans_col < M)
        matrix_trans_dev[matrix_trans_row * M + matrix_trans_col] = buffer[threadIdx.x][threadIdx.y]; // 写入连续
}
 

void launch_matrix_trans_read_write_coalesced(const float *matrix_dev, float *matrix_trans_dev, int M, int N){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((N + TileWidth - 1) / TileWidth, (M + TileWidth - 1) / TileWidth); 
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matrix_trans_read_write_coalesced<<<grid, block>>>(matrix_dev, matrix_trans_dev, M, N); 
    cudaDeviceSynchronize();
}