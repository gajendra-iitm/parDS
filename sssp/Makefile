all: data
	nvcc -O3 sssp.cu -o sssp.out -arch=native
	./sssp.out inputs/USA-road-d.NY.egr 

data:
	mkdir -p inputs
	wget -nc https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/USA-road-d.NY.egr -P inputs/
