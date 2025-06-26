#define TileWidth 16

__global__ void matmul_kernel_base(const float *A, const float *B, float *result, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i< K; i++){
        if(row < M && col < N)
            result[row * N + col] += A[row * K + i] * B[col * K + i];
    }
    
}

void launch_matmul(const float *A, const float *B, float *result, int M, int N, int K){
    dim3 block(TileWidth, TileWidth);
    dim3 grid((M + TileWidth - 1) / TileWidth, (N + TileWidth - 1) / TileWidth);
    matmul_kernel_base<<<grid, block>>>(A, B, result, M, N, K);
    cudaDeviceSynchronize();
}