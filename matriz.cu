#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error CUDA en linea " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

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
    int totalElementos = filas * columnas;
    size_t bytes = totalElementos * sizeof(float);

    // Memoria Host (CPU)
    float *h_A = new float[totalElementos];
    float *h_B = new float[totalElementos];
    float *h_C = new float[totalElementos];

    for (int i = 0; i < totalElementos; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    // Memoria Device (GPU)
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copia CPU -> GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configuración de Hilos y Bloques
    dim3 hilosPorBloque(16, 16);
    dim3 bloquesEnGrilla((columnas + 15) / 16, (filas + 15) / 16);

    // Lanzamiento del Kernel
    sumarMatricesGPU<<<bloquesEnGrilla, hilosPorBloque>>>(d_A, d_B, d_C, filas, columnas);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copia GPU -> CPU
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "[+] Exito en la operacion paralelo en GPU!" << std::endl;
    std::cout << "Resultado C[0]: " << h_C[0] << " (Esperado: 3)" << std::endl;
    std::cout << "Resultado C[999999]: " << h_C[totalElementos - 1] << " (Esperado: 3)" << std::endl;

    // Liberación de memoria
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}