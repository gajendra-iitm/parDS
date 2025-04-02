#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <chrono>

// #define N	1048576 //1073741824 //2^30     // 1048576 //2^20		// must be a power of 2. 
#define BLOCKSIZE	256 

__global__ void reduce2(int *input, int N) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int off = N / 2; off; off /= 2) {
    if (id < off)
      input[id] += input[id + off];
    __syncthreads();
  }
}

__global__ void reduce4(int *input , int N) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int off = N / 2; off; off /= 2) {
		if (id < off)
			input[id] += input[2 * off - id - 1];
		__syncthreads();
	}
}

__global__ void reduce5(int *input, int N) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int off = N / 2; off; off /= 2) {
		if (id < off)
			input[N / off * id] += input[N / off * id + N / 2 / off];
		__syncthreads();
	}
}

__global__ void reduce3(int *input, int *result, int N) {
    __shared__ unsigned sdata[BLOCKSIZE];  // Shared memory for block reduction

    unsigned tid = threadIdx.x; 
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load global memory into shared memory (handle out-of-bounds threads)
    sdata[tid] = (id < N) ? input[id] : 0;
    __syncthreads();

    //Perform reduction in shared memory
      for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
          if (tid < offset) {
              sdata[tid] += sdata[tid + offset];
          }
          __syncthreads();  // Synchronize threads within block
      }
    // for (int offset = 1, end = blockDim.x ; offset < end; offset = offset *2) {
        // if (tid < offset) {
            // sdata[tid] += sdata[tid + offset];
        // }
        // __syncthreads();  // Synchronize threads within block
    // }

    // Only one thread per block adds the result atomically to global sum
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


int main(int argc, char **argv) {
  int k = 20;
  if(argc > 1) k = atoi(argv[1]);
  const int N = 1 << k;

	int hinput[N];
	int cpu_sum = 0;
   
	// Initialize array and calculate CPU sum
	for (unsigned ii = 0; ii < N; ++ii) {
		hinput[ii] = rand() % 20;
	}
	
	// Time CPU sum calculation
	auto start = std::chrono::high_resolution_clock::now();
  for (unsigned ii = 0; ii < N; ++ii) {
		cpu_sum += hinput[ii];
	}
	
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  //std::cout << "seqSum:" << cpu_sum << ",Time:" << elapsed.count() << " seconds" << std::endl;
  printf("N=2^%d seqSum | Sum: %u |  GPU Time: %f ms\n", k,cpu_sum,  elapsed.count()*1000);
  
	int nblocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
	int *input;
	int *result;
  
  cudaMalloc(&input, N * sizeof(int));
	cudaMalloc(&result, sizeof(int));

	float gpu_time;
	int gpu_sum; 
    
	// Create CUDA events for timing
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


	// RKPlusNBy2 reduce2 kernel

	cudaMemcpy(input, hinput, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(gpu_start);
	
  reduce2<<<nblocks, BLOCKSIZE>>>(input,N);
  cudaMemcpy(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost);
  
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaMemcpy(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("N=2^%d RKPlusNBy2_reduce2 | Sum: %u |  GPU Time: %.3f ms\n", k,gpu_sum,  gpu_time);

	// RKNminusI kernel
	
	cudaMemcpy(input, hinput, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(gpu_start);
	reduce4<<<nblocks, BLOCKSIZE>>>(input,N);
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaMemcpy(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("N=2^%d RKNminusI-reduce4 | Sum: %u |  GPU Time: %.3f ms\n", k,gpu_sum,  gpu_time);

	// RKConsecutive kernel
	
	cudaMemcpy(input, hinput, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(gpu_start);
	reduce5<<<nblocks, BLOCKSIZE>>>(input,N);
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaMemcpy(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("N=2^%d RKConsecutive-reduce5 | Sum: %u GPU Time: %.3f ms\n", k,gpu_sum,  gpu_time);

	// RKPlusNBy2_MODIFIED -red3 kernel
	
	cudaMemcpy(input, hinput, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(gpu_start);
 
  reduce3<<<nblocks, BLOCKSIZE>>>(input,result,N);
  
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaMemcpy(&gpu_sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("N=2^%d RKPlusNBy2_MODIFIED | Sum: %u GPU Time: %.3f ms\n", k,gpu_sum,  gpu_time);

	// Cleanup
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	cudaFree(input);

	return 0;
}
