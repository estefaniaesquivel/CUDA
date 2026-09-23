#ifndef __INTELLISENSE__
#include <iostream>
#include <cuda_runtime.h>


__global__ void sumarMatricesGPU(float* A, float* B, float* C, int filas, int columnas) {
   
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;


    if (fila < filas && col < columnas) {
        int idx = fila * columnas + col; 
        C[idx] = A[idx] + B[idx];       
    }
}

int main() {
    int filas = 1000;
    int columnas = 1000;
    size_t bytes = filas * columnas * sizeof(float);

   
    float *h_A = new float[filas * columnas];
    float *h_B = new float[filas * columnas];
    float *h_C = new float[filas * columnas];

  
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    
    dim3 hilosPorBloque(16, 16);
    dim3 bloquesEnGrilla((columnas + 15) / 16, (filas + 15) / 16);

    
    sumarMatricesGPU<<<bloquesEnGrilla, hilosPorBloque>>>(d_A, d_B, d_C, filas, columnas);

    
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
#endif // __INTELLISENSE__