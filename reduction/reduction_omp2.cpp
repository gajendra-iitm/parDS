// g++ -Wall -o "reduction_omp2.out" "reduction_omp2.cpp"  -O3 -fopenmp
#include <omp.h>

#include <chrono>
#include <iostream>

template <typename T>
T parallelReduce1(T *input, int N) {
  T sum = 0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }

  return sum;
}

// template <typename T>
long long parallelReduce1(int *input, int N) {
  long long sum = 0;
// short PARLIMIT = 1000;
#pragma omp parallel for reduction(+ : sum)  // num_threads(PARLIMIT)
  for (int i = 0; i < N; /*i+=PARLIMIT*/ ++i) {
    sum += input[i];
  }
  return sum;
}

// template <typename T>
long long seqReduceSum(int *input, int N) {
  long long sum = 0;
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }

  return sum;
}
void randomInputs(int *input, int N) {
  srand(time(0));
  for (int i = 0; i < N; i++) {
    input[i] = static_cast<float>(rand() % 100 + 1);
  }
}
int main(int argc, char **argv) {
  int pow = 20;
  int N = 1 << pow;
  if (argc > 1) {
    pow = atoi(argv[1]);
    N = 1 << pow;
  }
  // std::cout<< "N: " << N  << std::endl;
  // int *input = new int[N]; //seqfault for 2^30.
  int *input = (int *)malloc(sizeof(int) * N); // still seq faults
  randomInputs(input, N);

  std::cout << "pow:2^x,parSum,Time(s),seqSum,Time(s)" << std::endl;
  for (int ii = 10; ii < pow+1; ii += 5) {
    N = 1 << ii;
    
    //PARREDUCE
    auto start = std::chrono::high_resolution_clock::now();
    long long int sum = parallelReduce1(input, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "2^" << ii << "," << sum << "," << elapsed.count() << ",";

    // SEQREDUCE
    start = std::chrono::high_resolution_clock::now();
    sum = seqReduceSum(input, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "" << sum << "," << elapsed.count() << " s" ;

    std::cout << std::endl;
  }
  // delete[] input;
  free(input);
  return 0;
}

