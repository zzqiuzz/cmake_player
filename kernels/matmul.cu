#define TileWidth 32 
#define TileKWidth 32 // 32
#define TileMWidth 32 
#define TileNWidth 32 
#include <iostream>
using std::cout;
using std::endl;
__global__ void matmul_kernel_base(const float *A, const float *B, float *result, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i< K; i++){
        if(row < M && col < N)
            result[row * N + col] += A[row * K + i] * B[i * N + col];
    }
    
}

void launch_matmul_naive(const float *A, const float *B, float *result, int M, int N, int K){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((N + TileWidth - 1) / TileWidth, (M + TileWidth - 1) / TileWidth);
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matmul_kernel_base<<<grid, block>>>(A, B, result, M, N, K);
    cudaDeviceSynchronize();
}



__global__ void matmul_kernel_tile(const float *A, const float *B, float *result, int M, int N, int K) {
    // define shared memory  equals to block shape
    __shared__ float subTiledA[TileMWidth][TileKWidth]; // 32 x 16
    __shared__ float subTiledB[TileKWidth][TileNWidth]; // 16 x 32

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    float resultValue = 0.f;
    for(int m = 0; m < (K + TileKWidth - 1) / TileKWidth; m++){
        // fill data in subTiledA and B
        int a_row = row;
        int a_col = m * TileKWidth + threadIdx.x;
        if(threadIdx.x < TileKWidth){
            if(a_row < M && a_col < K)
                subTiledA[threadIdx.y][threadIdx.x] = A[row * K + m * TileKWidth + threadIdx.x];    //A[row][m * TileWidth + threadIdx.x]; 
            else
                subTiledA[threadIdx.y][threadIdx.x] = 0.f;
        }

        int b_row = m * TileKWidth + threadIdx.y;
        int b_col = col;
        if(threadIdx.y < TileKWidth){
            if(b_row < K && b_col < N)
                // subTiledB[threadIdx.y][threadIdx.x] = B[col * K + m * TileKWidth + threadIdx.y];  error
                subTiledB[threadIdx.y][threadIdx.x] = B[b_row * N + b_col]; // 行优先的话，按照行来索引！
            else    
                subTiledB[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads(); // wait all threads in this block  finish loading data from global memory
        // calc value
        for(int i = 0; i < TileKWidth; i++){
            resultValue += subTiledA[threadIdx.y][i] * subTiledB[i][threadIdx.x];
        }
        __syncthreads(); // wait all the m-th subtile cal done
    } 
    if(row < M && col < N)
        result[row * N + col] = resultValue; 
     
    
}

void launch_matmul_tiled(const float *A, const float *B, float *result, int M, int N, int K){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((N + TileWidth - 1) / TileWidth, (M + TileWidth - 1) / TileWidth); 
    cout << "block.x = " << block.x << " block.y = " << block.y << " block.z = " << block.z << endl;
    cout << "grid.x = " << grid.x << " grid.y= " << grid.y << " grid.z = " << grid.z;
    cout << endl;
    matmul_kernel_tile<<<grid, block>>>(A, B, result, M, N, K);
    cudaDeviceSynchronize();
}