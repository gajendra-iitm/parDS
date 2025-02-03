#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void reduce0(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < g_idata[0]) ? g_idata[i + 1] : 0; // Adjusting for size
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < g_idata[0]) ? g_idata[i + 1] : 0; // Adjusting for size
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv) {
    int N = 10; // Default size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Array size must be a positive integer." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Allocate host memory
    int *h_idata = new int[N + 1]; // +1 for the first element as size
    int *h_odata0 = new int[(N + 255) / 256]; // Number of blocks for reduce0
    int *h_odata1 = new int[(N + 255) / 256]; // Number of blocks for reduce1

    // Initialize random number generator
    std::srand(static_cast<unsigned>(std::time(0)));

    // Fill the input array with random numbers and store the size at index 0
    h_idata[0] = N; // Store size in the first element
    for (int i = 1; i <= N; ++i) {
        h_idata[i] = std::rand() % 100; // Random numbers between 0 and 99
        //std::cout << h_idata[i] << " "; // Print random numbers for reference
    }
    std::cout << std::endl;

    // Allocate device memory
    int *d_idata, *d_odata0, *d_odata1;
    cudaMalloc(&d_idata, (N + 1) * sizeof(int));
    cudaMalloc(&d_odata0, ((N + 255) / 256) * sizeof(int));
    cudaMalloc(&d_odata1, ((N + 255) / 256) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_idata, h_idata, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch reduce0 kernel and measure time
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    reduce0<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_idata, d_odata0);
    
    cudaEventRecord(stop);
    
    cudaMemcpy(h_odata0, d_odata0, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Final reduction on host for reduce0
    int final_sum_0 = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        final_sum_0 += h_odata0[i];
    }

   std::cout << "Kernel reduce0 - Final sum: " << final_sum_0 << ", Time: " << milliseconds << " ms" << std::endl;

   // Launch reduce1 kernel and measure time
   cudaMemcpy(d_idata, h_idata, (N + 1) * sizeof(int), cudaMemcpyHostToDevice); // Reset input data

   cudaEventRecord(start);
   
   reduce1<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_idata, d_odata1);
   
   cudaEventRecord(stop);
   
   cudaMemcpy(h_odata1, d_odata1, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
   
   cudaEventSynchronize(stop);
   
   cudaEventElapsedTime(&milliseconds, start, stop);

   // Final reduction on host for reduce1
   int final_sum_1 = 0;
   for (int i = 0; i < blocksPerGrid; ++i) {
       final_sum_1 += h_odata1[i];
   }

   std::cout << "Kernel reduce1 - Final sum: " << final_sum_1 << ", Time: " << milliseconds << " ms" << std::endl;

   // Clean up
   delete[] h_idata;
   delete[] h_odata0;
   delete[] h_odata1;

   cudaFree(d_idata);
   cudaFree(d_odata0);
   cudaFree(d_odata1);

   return EXIT_SUCCESS;
}
