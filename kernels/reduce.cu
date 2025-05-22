
__global__ void reduce0(float* device_input_data, float* device_out){


    float * input_data_begin_per_block = device_input_data + blockIdx.x * blockDim.x; 


    for(int i = 1; i < blockDim.x; i *= 2){
        if(threadIdx.x % (2 * i) == 0){
            input_data_begin_per_block[threadIdx.x] += input_data_begin_per_block[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
        device_out[blockIdx.x] = input_data_begin_per_block[0];
    // if(tid % 2 == 0){ 0 2 4 6
    //     device_input_data[tid] += device_input_data[tid + 1];
    // }
    // if(tid % 2 == 4){ 0 4
    //     device_input_data[tid] += device_input_data[tid + 2];
    // if(tid % 2 == 4){ 0
    //     device_input_data[tid] += device_input_data[tid + 2];
    

}





void reduce_baseline(float *device_input_data, float *device_out, int ThreadPerBlock, int N){
    dim3 block(ThreadPerBlock, 1, 1);   
    dim3 grid((N + ThreadPerBlock - 1) / ThreadPerBlock, 1, 1); // Calculate the number of blocks needed

    reduce0<<<grid, block>>>(device_input_data, device_out); // Launch the kernel
    cudaDeviceSynchronize();


}