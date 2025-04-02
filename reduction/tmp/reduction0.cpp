#include <chrono>
#include <iostream>
using namespace std;
void randomize(int* arr) {
:
}
int main(int argc, char** argv) {
  int n = 1000;
  if (argc == 2) {
    n = atoi(argv[0]);
  }
  // cout << n << endl;
  int* arr = (int*)malloc(n * sizeof(int));
  randomize(arr);
  long long int sum = 0;  // IDENTITY of OPER

  auto start = std::chrono::high_resolution_clock::now();
  for (int ii = 0; ii < n; ++ii) sum = sum + arr[ii];
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  cout << "sum:" << sum << TIME : " << elapsed.count() << endl;
}
