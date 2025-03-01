all: cuda-map.cu omp-map.cpp
	nvcc -O3 cuda-map.cu -o cuda-map
	g++ -O3 -fopenmp omp-map.cpp -o omp-map

clean:
	rm cuda-map omp-map
