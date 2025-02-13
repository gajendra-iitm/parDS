#include <iostream>
#include <omp.h>
#include <chrono>

float parallelReduceSum(float *input, int N) {
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    
    return sum;
}

int main() {
    int N = 1 << 20;
    float *input = new float[N];
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 1000 + 1);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    float sum = parallelReduceSum(input, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Reduced Sum: " << sum << std::endl;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    
    delete[] input;
    return 0;
}
