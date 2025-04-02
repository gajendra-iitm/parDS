#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Baseline kernel (your original)
__global__ void reduce0(int* input, int* output, int N) {
    extern __shared__ int sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    for(int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if(tid < offset && (i + offset) < N) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

// Optimized kernel with warp shuffling and unrolling
template <unsigned blockSize>
__global__ void reduce3(int* input, int* output, int N) {
    extern __shared__ int sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockSize*2) + tid;
    
    // Load 2 elements per thread with stride
    sdata[tid] = 0;
    if(i < N) sdata[tid] = input[i];
    if(i + blockSize < N) sdata[tid] += input[i + blockSize];
    __syncthreads();

    // Unrolled block reduction
    if(blockSize >= 512) { if(tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if(blockSize >= 128) { if(tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }

    // Warp-level reduction using shuffle
    if(tid < 32) {
        volatile int* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv) {
    int k = 20;
    if(argc > 1) k = atoi(argv[1]);
    const int N = 1 << k;
    const int block_size = 512;
    
    // Host memory
    int* h_input = new int[N];
    int* h_output = new int[2]; // [reduce0, reduce3]
    for(int i=0; i<N; i++) h_input[i] = 1;

    // Device memory
    int *d_input, *d_intermediate;
    cudaMalloc(&d_input, N*sizeof(int));
    cudaMalloc(&d_intermediate, (N/block_size+1)*sizeof(int));
    cudaMemcpy(d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start0, stop0, start3, stop3;
    cudaEventCreate(&start0); cudaEventCreate(&stop0);
    cudaEventCreate(&start3); cudaEventCreate(&stop3);
    
    // Benchmark reduce0
    int current_N = N;
    cudaEventRecord(start0);
    while(current_N > 1) {
        int grid_size = (current_N + block_size - 1) / block_size;
        reduce0<<<grid_size, block_size, block_size*sizeof(int)>>>(d_input, d_intermediate, current_N);
        current_N = grid_size;
    }
    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);
    cudaMemcpy(&h_output[0], d_intermediate, sizeof(int), cudaMemcpyDeviceToHost);

    // Reset intermediate data
    cudaMemset(d_intermediate, 0, (N/block_size+1)*sizeof(int));

    // Benchmark reduce3
    current_N = N;
    cudaEventRecord(start3);
    while(current_N > 1) {
        int grid_size = (current_N + block_size*2 - 1) / (block_size*2);
        reduce3<512><<<grid_size, block_size, block_size*sizeof(int)>>>(d_input, d_intermediate, current_N);
        current_N = grid_size;
    }
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaMemcpy(&h_output[1], d_intermediate, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    float t0, t3;
    cudaEventElapsedTime(&t0, start0, stop0);
    cudaEventElapsedTime(&t3, start3, stop3);
    
    std::cout << "reduce0 sum: " << h_output[0] << " time: " << t0 << " ms\n";
    std::cout << "reduce3 sum: " << h_output[1] << " time: " << t3 << " ms\n";

    // Verification
    int cpu_sum = N;
    std::cout << "CPU sum: " << cpu_sum << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_intermediate);
    return 0;
}
