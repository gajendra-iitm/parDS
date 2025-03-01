all: cuda-heap.cu omp-heap.cpp
	nvcc -O3 cuda-heap.cu -o cuda-heap
	g++ -O3 -fopenmp omp-heap.cpp -o omp-heap

clean:
	rm cuda-heap omp-heap
