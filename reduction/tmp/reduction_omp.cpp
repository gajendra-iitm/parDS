#include <iostream>
#include <omp.h>
#include <chrono>
#include <fmt/core.h>

template <typename T>
T parallelReduceSum(T *input, int N) {
    T sum = 0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    
    return sum;
}

template <typename T>
T seqReduceSum(T *input, int N) {
    T sum = 0;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    
    return sum;
}
int main() {
    int N =  1 << 20;
    int *input = new int[N];
    srand(time(0));
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand()% 100 + 1);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    float sum = parallelReduceSum(input, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    //std::cout << "Reduced Sum: " << sum << std::endl;
    std::cout << "parSum:" << sum << ",Time:" << elapsed.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    sum = seqReduceSum(input, N);
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "seqSum:" << sum << ",Time:" << elapsed.count() << " seconds" << std::endl;
    
    delete[] input;
    return 0;
}
    
