long long parallelReduce1(int *input, int N) {
  long long sum = 0;
  #pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }
  return sum;
}
