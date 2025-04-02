#include <stdio.h>
#include <cuda.h>

#define N		1048576		// must be a power of 2.
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

	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	RKPlusNBy2<<<nblocks, BLOCKSIZE>>>(nelements);

	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	RKNminusI<<<nblocks, BLOCKSIZE>>>(nelements);

	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);
	RKConsecutive<<<nblocks, BLOCKSIZE>>>(nelements);

	cudaDeviceSynchronize();

	return 0;
}
