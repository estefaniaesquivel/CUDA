#include <iostream>

__global__ void testKernel() {}

int main() {
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "CUDA OK!" << std::endl;
    return 0;
}