// CSR Bellman ford 
#include <algorithm>
#include <iostream>
#include <map>
#include <stack>
#include <sstream>
#include <climits>
#include <vector>
#include <set>

//~ #include <signal.h>
//~ #include <unistd.h> // No SIGKILL as of now in GPU!

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 }                                                                 \
}

#include <cstring>
#include <cmath>

#include <cuda.h>
#include <stdio.h>

using namespace std;

//~ #define LEVEL 0 // 1 - print all 0 -- submission level
int LEVEL = 0;

#define DEBUG if(LEVEL) 

#define DELIMITER INT_MIN

class Edge { 
	
public:
	int to;
	int length; 
	
	Edge(){}
	~Edge(){}
	Edge(int t, int l){
		to = t; length = l;
	}
	bool operator < (const Edge& e){
		return length < e.length;
	}
};
/*
 * 
 * KERNAL edge list
 *  N - #nodes
 *  M - #edges
 *  src - source to start the SSSP
 *  edge list or adj list
 */ 
__global__ void kernelBellmanFordMoore(int N, int M, int src, 
	int* e1, int* e2, int* e1W, int* e2W, int *changed,
	int *minDist, int *parent){
	
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(id < M){	
		int u = e1[id];
		int v = e2[id];
		int w = e1W[id];	
		
		int newDist =  minDist[u]+w;
		int oldMinD = minDist[v];
		if(	newDist < minDist[v] )
			atomicMin(&minDist[v], newDist);
					
		//~ if(	newDist < minDist[v] ) { // the to perform relax!
			//~ minDist[v] = newDist;
		//~ }
		__syncthreads(); // is it really needed? yes!
		
		if(oldMinD!=minDist[v])
			changed[0] = 1;
		
		//~ Since undirected relaxing in both direction! - added rupesh discussion!
		u = e2[id]; 
		v = e1[id]; 
		newDist =  minDist[u]+w;
		if(	newDist < minDist[v] )
			atomicMin(&minDist[v], newDist);
	}	
}	

__global__ void csrKernelBellmanFordMoore(int N, int M, int _2M, int src, 
	int* csrM, int* csrD, int* csrW,
	int *changed,
	int *minDist, int *parent){
	
	//~ int end = _2M;
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(id < N){	// = is not need coz N is already n+1; 0 the processing unwanted but ok for now.
		
		int u = id;   // may not be needed but easier to code!
		int end= csrM[u+1];

		// for all its nbrs of id / ith node		
		for(int i=csrM[u]; i< end; i++){
			int v = csrD[i];
			/// relax
			int newDist =  minDist[u]+ csrW[i];
			int old = minDist[v];			
			//~ if (id == 53 && v==11) {
				//~ printf("oldvalue = %d, new value = %d\n",minDist[v], newDist);
			//~ }
			if(newDist < old){ // actually this not needed. This is what atomic does. But it is here avoid conjection of atomic stmts
				old=atomicMin(&minDist[v], newDist); 
				// do parent later!
				//~ printf("upd!\n");
				changed[0]= 1;
			}
			//~ __syncthreads(); -- NOT needed
			//~ if(newDist < old){
				//~ changed[0]= 1;
			//~ }
		}
	}
					
}

__global__ void kernelInit(int N, int src, int *minDist, int* parent){
	
	unsigned id = threadIdx.x + blockDim.x * blockIdx.x;	
  if(id > N) return;				

	minDist[id]=INT_MAX/2;
	parent[id] = -1;
	
	if(id == src){
		minDist[id]=0;		
		minDist[0]=0;
	}	
}
void computeCSR(const vector< vector<Edge> > &graph, int* csrM, int* csrD, int* csrW, int N){
	//~ cout << "Print ADJ"<< endl;
	 
	int idx = 0;
	// init for first two idx
	
	csrM[0]=0;
	csrM[1]=0;
	
	DEBUG printf("P[%d] = %d\n", 0, csrM[0]);
	
	//~ for (auto vec : graph){  // This line is working in v9.2 but not on 9.1 :(
	 
	for(int i=1, endI = graph.size(); i < endI; i++){	 
		//~ cout << i << " ";
		int endJ = graph[i].size();
		//~ cout << i << " : " << vec.size() << endl ;
		csrM[i+1] = csrM[i] + endJ; // as 0th node has nothing adjacent
		           
		DEBUG printf("P[%d] = %d\n", i, csrM[i]);
		
		//~ for(auto e : vec){ -- as above for comment
		for(int j=0 ; j < endJ; j++){
			csrD[idx] =  graph[i][j].to;
			csrW[idx++] = graph[i][j].length; // note! idx post inc
		}
	}
	//~ DEBUG printf("P[%d] = %d\n", graph.size(), csrM[graph.size()]);	
	DEBUG printf("P[%d] = %d idx=%d\n", (int)graph.size(), csrM[(int)graph.size()], idx);
	
}
/*
 * 
 * Print any host array of the size
 *  with a string s
 */
void printArray(string s, int *arr, int size, bool fromOne=false){
	cout << s << endl;
	
	for( int i=0  ; i < size; i++){
		cout <<"Ar[" <<i << "] = "<< arr[i] << endl;
	}
	
}

 int main(int argc, char *grgv[]){
	//~ ios_base::sync_with_stdio(false);
	
	if(argc > 1){
		LEVEL = 1;
	}
	
	vector< vector<Edge> > graph;
	map<pair<int,int> , int> W;
	vector < vector<int> > tdEdge;
	vector < vector<int> > tdBag;
	vector <int> terminals;
	set <int> terminalSet;
	string code, type, dummy;
	int N = 0, M = 0;
	
	// Host vars -- for input
	int* eList1;
	int* eList2;
	int* eList1W;
	int* eList2W;
	
	int* csrMeta_h;
	int* csrData_h;
	int* csrDataWt_h;
	
	//~ int eList1[M];
	//~ int eList2[M];
	//~ int eList1W[M];
	//~ int eList2W[M];
	 
	while( cin>> code >> type ){
		
		if(code == "SECTION" && type =="Graph"){
			long m, n;
			long u, v, w;
			cin >> dummy >> n;
			cin >> dummy >> m;
			N = n+1; 					// THIS IS IMPORTANT!!!
			M = m;
			//~ eList1 = (int*) malloc(M);
			//~ eList2 = (int*) malloc(M);
			//~ eList1W = (int*) malloc(M);
			//~ eList2W = (int*) malloc(M);
			eList1 = new int [M];  // nor required for csr kernel!
			eList2 = new int [M];
			eList1W = new int [M];
			eList2W = new int [M];
			
			csrMeta_h = new int [N+1]; // as meta is more than!! N
			csrData_h = new int [2*M];
			csrDataWt_h = new int [2*M];
			
			DEBUG cout << "N=" << N <<" M="<< M  << endl;
			graph.resize(N); // coz graph has from index 0. where as challege its 1
			for(long i=0; i < m; i++){
				cin>> dummy >> u >> v >> w;
				//~ cout <<  u<< " -- " << v << " : " << w<< endl;
				graph[u].push_back(Edge(v,w));
				graph[v].push_back(Edge(u,w));
				//~ W[make_pair(u,v)]=w;
				//~ W[make_pair(v,u)]=w;
				eList1[i]= u;
				eList2[i]= v;
				eList1W[i]= w;
				eList2W[i]= w;
			}
			cin >> dummy >> ws;
		}
		else if(code == "SECTION" && type =="Terminals"){
			long t, u;
			cin >> dummy >> t;
			for(long i=0; i < t; i++){
				cin>> dummy >> u;
				terminals.push_back(u);
				terminalSet.insert(terminalSet.end(), u);
			}
			cin >> dummy >> ws;
		}
		else if(code == "SECTION" && type =="Tree"){
			
			cin >> dummy >> ws; // DECOMP
			cin >> dummy >> dummy; // s ,td
			long bags, bid, val ; 
			cin >> bags; // total bags
			tdEdge.resize(bags+1);
			tdBag.resize(bags+1);
			cin >> dummy >> dummy >> ws; // tw, n
			
			for(long i=0; i < bags; i++){
				string line;
				getline(cin, line); stringstream sstream(line);
				if(sstream >> dummy, dummy=="b"){
					sstream >> bid; // bag id
					//~ cout << bid << ": ";
					while(sstream >> val){
						//~ cout << val << " " ;
						tdBag[bid].push_back(val);
					}
					//~ cout << endl;
				}
			}
			long tu, tv;
			for(long i=1; i < bags; i++){ // b-1 edges is Td
				cin >>  tu >> tv;
				tdEdge[tu].push_back(tv);
				tdEdge[tv].push_back(tu);
			}
			cin >> dummy >> ws; // END
			cin >> dummy >> ws; // eof
			
		}
		else{
			cout << "Err in Input/Parsing!" << endl;
			exit(1);
		}
	
	}
	DEBUG cout << "Parsing done"<< endl;
	 // convert g to csr
	computeCSR(graph, csrMeta_h, csrData_h, csrDataWt_h, N);
	
	DEBUG cout << "CSR done"<< endl;
	
	//~ printArray("meta array",csrMeta_h, 2*M);
	//~ DEBUG printArray("meta array",csrData_h, 2*M);
	 
	
	//~ for(int i=0; i !=M ; i++)
		//~ cout << eList1[i]<< " == " << eList2[i] << " :: " << eList1W[i]<< endl;

	int *minDist; 
	int *parent; 
	
	//~ int *e1; int *e2;	
	//~ int *e1W; int *e2W;
	
	// inputs for csr 
	int *csrMeta_d; 
	int *csrData_d;
	int *csrDataWt_d;
	
	unsigned nSize = (sizeof(int)*(N));
	unsigned nSizePlus1 = (sizeof(int)*(N+1));
	//~ unsigned mSize = (sizeof(int)*(M));
	unsigned m2Size = (sizeof(int)*(2*M));
	
	// inputs allocations
	cudaMalloc((void**) &csrMeta_d, nSizePlus1); //
	cudaMalloc((void**) &csrData_d, m2Size);		
	cudaMalloc((void**) &csrDataWt_d, m2Size);		
	
	//~ cudaMalloc((void**) &e1, mSize);	
	//~ cudaMalloc((void**) &e2, mSize);	
	//~ cudaMalloc((void**) &e1W, mSize);	
	//~ cudaMalloc((void**) &e2W, mSize);	// Not needed?
		
	cudaMalloc((void**) &minDist, nSize);	
	cudaMalloc((void**) &parent, nSize);	// Not in use NOW!
	
	// HOST TO DEVICE -- inputs
	cudaMemcpy(csrMeta_d, csrMeta_h, nSize, cudaMemcpyHostToDevice);
	cudaMemcpy(csrData_d, csrData_h, m2Size, cudaMemcpyHostToDevice);
	cudaMemcpy(csrDataWt_d, csrDataWt_h, m2Size, cudaMemcpyHostToDevice);
	
	//~ cudaMemcpy(e1, eList1, mSize, cudaMemcpyHostToDevice);
	//~ cudaMemcpy(e2, eList2, mSize, cudaMemcpyHostToDevice);
	//~ cudaMemcpy(e1W, eList1W, mSize, cudaMemcpyHostToDevice);
	//~ cudaMemcpy(e2W, eList2W, mSize, cudaMemcpyHostToDevice);
	 
	unsigned threadInBlock = 1024;
	int src = 1; // soource in 1 hardcoding
	
	/* Lauch should be Threads=1024
	 * (N+1)/Threads, Threads
	 * 
	 */ 
	kernelInit<<< (N+(threadInBlock-1))/threadInBlock, min(N,threadInBlock)  >>>( N, src, minDist,parent);
	
	cudaCheckError();
	
	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
	
	//~ cudaEventRecord(start);
	
	int *changed;
	
	cudaHostAlloc(&changed, sizeof(int), 0);

	changed[0] =1; // for 1st for iter to be true!
	
	int _2M = M *2;
	
	//~ printf("N=%d  BLOCKs=%d THRES/BLOK=%d \n", N, (M+(threadInBlock-1))/threadInBlock, min(M,threadInBlock) );
	for(int i=1; i != (N-1) && changed[0]==1 ; i++){
		
		changed[0] = 0; // resetting the change flag
		//~ printf("RUN %d BEGIN; changed=%d \n",i, changed[0]);
		//~ printf("\tRUN %d BEGIN; changed=%d \n",i, changed[0]);
		csrKernelBellmanFordMoore<<<(N+(threadInBlock-1))/threadInBlock, min(N,threadInBlock)  >>>( N, M, _2M, src, 
		csrMeta_d, csrData_d, csrDataWt_d,  // inputs 
		changed, // fixed pt var
		minDist, parent); //these are outputs
		//~ cudaCheckError();
		
		cudaDeviceSynchronize();  //+
		//~ cudaCheckError();
		//~ printf("\tRUN %d END;   changed=%d \n",i, changed[0]);
		//~ cout << i << " ***********"<< endl;
	}
	//~ cudaEventRecord(stop);		
	//~ cudaEventSynchronize(stop);
	//~ float milliseconds = 0;
	//~ cudaEventElapsedTime(&milliseconds, start, stop);
	//~ printf("TIME GPU %f",milliseconds);
						
	cudaCheckError();
  
	// Copy output from device to HOST and print!
	int minD[N];
	cudaMemcpy(minD, minDist, nSize , cudaMemcpyDeviceToHost);
	
	for(int i=1; i < N; i++)
		//~ printf("minD[%d]=%d\n",i,minD[i]);
		cout << minD[i] << "\n";
	
	// cleaning up!	
	//~ cudaFree(e1);
	//~ cudaFree(e2);
	//~ cudaFree(e1W);
	//~ cudaFree(e2W);
	
	cudaFree(csrMeta_d);
	cudaFree(csrData_d);
	cudaFree(csrDataWt_d);
	
	cudaFree(parent);
	cudaFree(minDist);
	
	
	//~ cudaDeviceReset();
	  
	DEBUG cout << "RETURN!" << endl;
	return 0;
}
/* - Don't keep data in PINNED Memory!
 * - Keep the graph and output on GPU memory! 
 * - User an edge representation! 
 */ 
 
