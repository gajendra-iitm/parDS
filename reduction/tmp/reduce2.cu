#include <stdio.h>
#include <cuda_runtime.h>

#define N		1048576		// must be a power of 2
#define BLOCKSIZE	N

__global__ void RKPlusNBy2(unsigned *nelements) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int off = N / 2; off; off /= 2) {
		if (id < off)
			nelements[id] += nelements[id + off];
		__syncthreads();
	}
	if (id == 0)
		printf("GPU sum = %d\n", *nelements);
}

__global__ void RKNminusI(unsigned *nelements) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int off = N / 2; off; off /= 2) {
		if (id < off)
			nelements[id] += nelements[2 * off - id - 1];
		__syncthreads();
	}
	if (id == 0)
		printf("GPU sum = %d\n", *nelements);
}

__global__ void RKConsecutive(unsigned *nelements) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int off = N / 2; off; off /= 2) {
		if (id < off)
			nelements[N / off * id] += nelements[N / off * id + N / 2 / off];
		__syncthreads();
	}
	if (id == 0)
		printf("GPU sum = %d\n", *nelements);
}

int main() {
	unsigned hnelements[N];
	unsigned sum = 0;
	for (unsigned ii = 0; ii < N; ++ii) {
		hnelements[ii] = rand() % 20;
		sum += hnelements[ii];
	}
	printf("CPU sum = %d\n", sum);

	unsigned nblocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

	unsigned *nelements;
	cudaMalloc(&nelements, N * sizeof(unsigned));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float kernel_time;

	// Time RKPlusNBy2 kernel
	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	RKPlusNBy2<<<nblocks, BLOCKSIZE>>>(nelements);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);
	printf("RKPlusNBy2 execution time: %.3f ms\n\n", kernel_time);

	// Time RKNminusI kernel
	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	RKNminusI<<<nblocks, BLOCKSIZE>>>(nelements);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);
	printf("RKNminusI execution time: %.3f ms\n\n", kernel_time);

	// Time RKConsecutive kernel
	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	RKConsecutive<<<nblocks, BLOCKSIZE>>>(nelements);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);
	printf("RKConsecutive execution time: %.3f ms\n\n", kernel_time);

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(nelements);
	cudaDeviceSynchronize();

	return 0;
}
