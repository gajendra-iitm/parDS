// nvcc -o "sssp.out" "sssp.cu" -O3 -arch=native

#include <climits>
#include <algorithm>
#include <tuple>
#include <vector>
#include <set>
#include <sys/time.h>

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono>


#include "ECLgraph.h"

using namespace std;

static const int Device = 0;
static const int ThreadsPerBlock = 512;

// Moving to Chrono
struct CPUTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() { beg = std::chrono::high_resolution_clock::now();}
  double stop() {end = std::chrono::high_resolution_clock::now(); std::chrono::duration<double> elapsed = end - beg; return elapsed.count();}
};


// static void CheckCuda(const int line)
// {
  // cudaError_t e;
  // cudaDeviceSynchronize();
  // if (cudaSuccess != (e = cudaGetLastError())) {
    // fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    // exit(-1);
  // }
// }



// SeqSSSP. Dijkstra
// int* dijkstra( const ECLgraph& g){
vector<int> dijkstra( const ECLgraph& g){
  // int* min_distance = new int [g.nodes];
  // int* parent = new int [g.nodes];
  // std::fill(min_distance, min_distance + g.edges, INT_MAX/2);
  // std::fill(parent, parent + g.edges, -1);
  vector <int> min_distance(g.nodes,INT_MAX/2);
  vector<int> parent(g.nodes, -1);	
  
  int source = 0;
	min_distance[source] = 0;
	set< pair<int,int> > active_vertices;
	active_vertices.insert( {0,source} );
	
  CPUTimer timer;
  timer.start();
  
	while (!active_vertices.empty()) {
		int where = active_vertices.begin()->second;
		
		active_vertices.erase( active_vertices.begin() );
		 for (int j = g.nindex[where]; j < g.nindex[where + 1]; j++) {
      int v = g.nlist[j];
      int w = g.eweight[j];
			auto newdist = min_distance[where] + w;
			if (newdist < min_distance[v]) {
				active_vertices.erase( { min_distance[v],v } );
				min_distance[v] = newdist;
				parent[v] = where;
				active_vertices.insert( { newdist, v } );
			}
		}
	}
  
  const double runtime = timer.stop();
  printf("Dijkstra SSSP. Host: %12.9f s\n", runtime);
  
	return min_distance;
}


// Fixed point style sssp
// static int* cpuSSSP(const ECLgraph& g){
static vector<int> cpuSSSP(const ECLgraph& g){
  // int* minDist = new int [g.nodes];
  // int* parent = new int [g.nodes];
  // std::fill(minDist, minDist + g.edges, INT_MAX/2);
  // std::fill(parent, parent + g.edges, -1);
  vector <int> minDist(g.nodes,INT_MAX/2);   // want int* (as above) but seqfaulting
  vector<int> parent(g.nodes, -1);	
  
  // initialize
  int src = 0;
  parent[src] = 0;
  minDist[src] = 0;
  
  CPUTimer timer;
  timer.start();
  
  bool modified;
  int k=0;
  do{
		modified = false;
		//cout << k++ <<" of " << g.nodes-1 <<endl;  // THIS OR BELOW
    k++;
    for (int u = 0; u < g.nodes; u++) {
      for (int j = g.nindex[u]; j < g.nindex[u + 1]; j++) {
				int w = g.eweight[j]; // edge weight of (u,v)
				int v = g.nlist[j];
        // printf("E(%d,%d)=%d\n",u,v,w);
				int newDist =  minDist[u]+w;
				if(	newDist < minDist[v] ){ // the to perform relax!
					minDist[v] = newDist;
					parent[v] = u; 
					modified=true;
          //printf("E(%d,%d)=%d  == %d < minDist[%d]=%d\n",u,v,w,newDist,v,minDist[v]);
				}
			}
		}
    
	}while(modified);
  cout << k++ <<" iterations of " << g.nodes-1 <<endl;
  const double runtime = timer.stop();
  printf("FixedpointSSSP. Host: %12.9f s\n", runtime);
  return minDist;
}

#include <omp.h>

static vector<int> cpuParSSSP(const ECLgraph& g) {
  vector<int> minDist(g.nodes, INT_MAX / 2);
  vector<int> parent(g.nodes, -1);

  int src = 0;
  parent[src] = 0;
  minDist[src] = 0;

  CPUTimer timer;
  timer.start();

  bool modified;
  int k = 0;

  do {
    modified = false;

    // OpenMP: declare shared/firstprivate variables
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < g.nodes; u++) {
      int local_modified = 0; // per-thread flag
      for (int j = g.nindex[u]; j < g.nindex[u + 1]; j++) {
        int w = g.eweight[j];
        int v = g.nlist[j];
        int newDist = minDist[u] + w;
        if (newDist < minDist[v]) {
          // atomic compare-and-swap style update
          #pragma omp critical
          {
            if (newDist < minDist[v]) {
              minDist[v] = newDist;
              parent[v] = u;
              modified = true;
            }
          }
        }
      }
    }

    k++;
  } while (modified);

  cout << k << " iterations of " << g.nodes - 1 << endl;
  const double runtime = timer.stop();
  printf("Fpt Par SSSP. Host: %12.9f s\n", runtime);

  return minDist;
}

/*

// INIT KERNEL
__global__ void kernelInit(int N, int src, int *minDist, int* parent){
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;	
  if(id >= N) return;				

	minDist[id]=INT_MAX/2;
	parent[id] = -1;
	
	if(id == src){
		minDist[id]=0;		
	}	
}

// FPT SSSP KERNEL. V3
__global__ void csrKernelBellmanFordMoore(int N, int M, const int *d_nindex, const int* d_nlist, const int* d_eweight,
	bool *modified,
	int *minDist, 
  int *parent // not req
  ){
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < N){	
		int u = id;   
		int end= d_nindex[u+1];
		for(int i=d_nindex[u]; i< end; i++){ 
			int v = d_nlist[i];
			/// relax
			int newDist =  minDist[u]+ d_eweight[i];
			if(newDist < minDist[v]){ //To avoid conjection of atomic stmts
				atomicMin(&minDist[v], newDist); 
				*modified= true;
			}
		}
	}
					
}

// SSSP V4
__global__ void csrKernelBellmanFordMoore2(int N, int M, const int *d_nindex, const int* d_nlist, const int* d_eweight,
	bool *modified,
	int *minDist, 
  int *parent // not req
  ){
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < N){	
		int u = id;   
		int end= d_nindex[u+1];
		for(int i=d_nindex[u]; i< end; i++){ 
			int v = d_nlist[i];
			/// relax
			int newDist =  minDist[u]+ d_eweight[i];
			if(newDist < minDist[v]){ //To avoid conjection of atomic stmts
				atomicMin(&minDist[v], newDist); 
				*modified= true;
			}
		}
    for(int i=d_nindex[u]; i< end; i++){ 
			int v = d_nlist[i];
			/// relax
			int newDist =  minDist[u]+ d_eweight[i];
			if(newDist < minDist[v]){ //To avoid conjection of atomic stmts
				atomicMin(&minDist[v], newDist); 
				*modified= true;
			}
		}
    for(int i=d_nindex[u]; i< end; i++){ 
			int v = d_nlist[i];
			/// relax
			int newDist =  minDist[u]+ d_eweight[i];
			if(newDist < minDist[v]){ //To avoid conjection of atomic stmts
				atomicMin(&minDist[v], newDist); 
				*modified= true;
			}
		}
	}
					
}


// GPU DRIVER SSSP
static int* gpuSSSP(const ECLgraph& g){
 
  // INPUTS
    int* d_nindex = NULL;
  cudaMalloc((void**)&d_nindex, (g.nodes + 1) * sizeof(int));
  cudaMemcpy(d_nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

  int* d_nlist = NULL;
  cudaMalloc((void**)&d_nlist, g.edges * sizeof(int));
  cudaMemcpy(d_nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice);

  int* d_eweight = NULL;
  cudaMalloc((void**)&d_eweight, g.edges * sizeof(int));
  cudaMemcpy(d_eweight, g.eweight, g.edges * sizeof(int), cudaMemcpyHostToDevice);
  
  // OUTPUTS
  
  int* d_minDist = nullptr;
  cudaMalloc((void**)&d_minDist, g.nodes * sizeof(int));
  int* const h_minDist = new int [g.nodes];
  
  // Optional Out
  int* d_parent = nullptr;
  cudaMalloc((void**)&d_parent, g.nodes * sizeof(int));
  
  // EXTRAS
  bool *d_modified;
  cudaMalloc((void**)&d_modified,   sizeof(bool));
  bool *modified = new bool[1];
  
  CPUTimer timer;
  timer.start();
  
  int src = 0;
  int k = 0;
  
  const int blocks = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
  kernelInit<<< blocks, ThreadsPerBlock >>>( g.nodes, src, d_minDist,d_parent);
	  
  
  do {
      modified[0] = false;
      cudaMemcpy(d_modified, modified, sizeof(bool), cudaMemcpyHostToDevice);
      csrKernelBellmanFordMoore2<<<blocks, ThreadsPerBlock>>>( g.nodes, g.edges, d_nindex, d_nlist, d_eweight, //inputs
                                                       d_modified,                        // fixed pt var
                                                       d_minDist, d_parent                // these are outputs
                                                       );

      

      cudaMemcpy(modified, d_modified, sizeof(bool), cudaMemcpyDeviceToHost);
      CheckCuda(__LINE__);      
      k++;
    } while (modified[0] == true);
  
  
  cudaDeviceSynchronize();
  const double runtime = timer.stop();
  printf("FPT GPU SSSP. DEV: %12.9f s\n", runtime);
   

  cudaMemcpy(h_minDist, d_minDist, g.nodes * sizeof(int), cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);
  
  
  cudaFree(d_minDist);
  cudaFree(d_parent);
  cudaFree(d_nindex);
  cudaFree(d_nlist);
  cudaFree(d_eweight);
  
  return h_minDist;
}
*/

// d1 and d2 vec of int
void verify0(const ECLgraph& g,  vector <int> &d1,vector <int> &d2){
// void verify1(const ECLgraph& g, int* d1, int* d2){
   int misMatch = 0;
  
  for (int j = 0; j < g.nodes; j++) {  // lets print only 10. //g.nodes
    if(d1[j] != d2[j]){
      misMatch++;
      printf("d1[%d]=%d  !=  d2[%d]=%d\n", j,d1[j],j,d2[j]);
    }
      
  }
    if (misMatch!=0) {
    printf("ERROR: results differ!\n\n");
  } else {
    printf("all good\n\n");
  }
} 
// d1 vec of int and d2 int*
void verify2(const ECLgraph& g, vector <int> &d1, int* d2){
   int misMatch = 0;
  
  for (int j = 0; j < g.nodes; j++) {  
    if(d1[j] != d2[j]){
      misMatch++;
      printf("d1[%d]=%d  !=  d2[%d]=%d\n", j,d1[j],j,d2[j]);
    }
    // if(j < 10)  
    // printf("d1[%d]=%d  !=  d2[%d]=%d\n", j,d1[j],j,d2[j]); // lets print only 10. //g.nodes
  }
    if (misMatch!=0) {
    printf("ERROR: results differ!\n\n");
  } else {
    printf("all good\n\n");
  }
} 

int main(int argc, char *argv[]){
  // process command line
  if (argc != 2) {printf("USAGE: %s input_file_name\n\n", argv[0]);  exit(-1);}

  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n\n", 1.0 * g.edges / g.nodes);

  // get GPU info
  // cudaSetDevice(Device);
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, Device);
  // if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  // const int SMs = deviceProp.multiProcessorCount;
  // const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  // printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);  fflush(stdout);
  //const int blocks = SMs * (mTpSM / ThreadsPerBlock);
  
  // Two diff CPU SSSP Implementation
  // int* dij_cpu_MinDist = cpuSSSP(g); // Seqfault
  vector<int> fpt_cpu_MinDist = cpuSSSP(g); 
  vector<int> omp_cpu_MinDist = cpuParSSSP(g);
  
  // GPU SSSP
  // int* gpu_MinDist = gpuSSSP(g);
  
  printf("dij == fpt\n");
  verify0(g,fpt_cpu_MinDist,omp_cpu_MinDist);
  
  // printf("fptcpu == fptgpu\n");
  // verify2(g,omp_cpu_MinDist,gpu_MinDist);
  
  freeECLgraph(g);
  return 0;
}
