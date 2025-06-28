#define TileWidth 16

__global__ void matmul_kernel_base(const float *A, const float *B, float *result, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i< K; i++){
        if(row < M && col < N)
            result[row * N + col] += A[row * K + i] * B[col * K + i];
    }
    
}

void launch_matmul_naive(const float *A, const float *B, float *result, int M, int N, int K){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((M + TileWidth - 1) / TileWidth, (N + TileWidth - 1) / TileWidth);
    matmul_kernel_base<<<grid, block>>>(A, B, result, M, N, K);
    cudaDeviceSynchronize();
}



__global__ void matmul_kernel_tile(const float *A, const float *B, float *result, int M, int N, int K) {
    // define shared memory  equals to block shape
    __shared__ float subTiledA[TileWidth][TileWidth];
    __shared__ float subTiledB[TileWidth][TileWidth];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    float resultValue = 0.f;
    for(int m = 0; m < K / TileWidth; m++){
        // fill data in subTiledA and B
        subTiledA[threadIdx.y][threadIdx.x] = A[row * K + m * TileWidth + threadIdx.x];    //A[row][m * TileWidth + threadIdx.x]; 
        subTiledB[threadIdx.y][threadIdx.x] = B[col * K + m * TileWidth + threadIdx.y]; 
        __syncthreads(); // wait all threads in this block  finish loading data from global memory
        // calc value
        for(int i = 0; i < TileWidth; i++){
            resultValue += subTiledA[threadIdx.y][m] * subTiledB[m][threadIdx.x];
        }
        __syncthreads(); // wait all the m-th subtile cal done
    } 
    result[row * N + col] = resultValue; 
     
    
}

void launch_matmul_tiled(const float *A, const float *B, float *result, int M, int N, int K){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((N + TileWidth - 1) / TileWidth, (M + TileWidth - 1) / TileWidth);
    matmul_kernel_tile<<<grid, block>>>(A, B, result, M, N, K);
    cudaDeviceSynchronize();
}