// nvcc -o "file.out" "file.cu" -O3

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Author: Rajesh Pandian M | mrprajesh.co.in
//  START: Thu,03-Apr-2025,03:48:39 IST
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include "ECLgraph.h"


static const int Device = 0;
static const int ThreadsPerBlock = 512;

typedef unsigned long long ull;

__global__ void k(){
  printf("hello %u!\n", threadIdx.x);
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
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);  fflush(stdout);
  const int blocks = SMs * (mTpSM / ThreadsPerBlock);
  
  
  return 0;
}
