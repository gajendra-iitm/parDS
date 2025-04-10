all: data
	g++ -O3 sssp_omp.cpp -o sssp_omp.out -fopenmp && ./sssp_omp.out inputs/USA-road-d.NY.egr
	nvcc -O3 sssp.cu -o sssp.out -arch=native &&	./sssp.out inputs/USA-road-d.NY.egr 

cuda:
	nvcc -O3 sssp.cu -o sssp.out -arch=native && ./sssp.out inputs/USA-road-d.NY.egr 

omp:
	g++ -O3 sssp_omp.cpp -o sssp_omp.out -fopenmp && ./sssp_omp.out inputs/USA-road-d.NY.egr

data:
	mkdir -p inputs
	wget -nc https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/USA-road-d.NY.egr -P inputs/
