#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

//nvcc matriz.cu -o matriz.exe
//.\matriz.exe
//.\matriz.exe > matriz.txt

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error CUDA en linea " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

__global__ void multiplicarMatricesGPU(const float* A, const float* B, float* C, int N) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N && col < N) {
        float suma = 0.0f;
        for (int k = 0; k < N; ++k) {
            suma += A[fila * N + k] * B[k * N + col];
        }
        C[fila * N + col] = suma;
    }
}

int main() {
    int N = 10000; 
    int totalElementos = N * N;
    size_t bytes = totalElementos * sizeof(float);

    float *h_A = new float[totalElementos];
    float *h_B = new float[totalElementos];
    float *h_C = new float[totalElementos];

    for (int i = 0; i < totalElementos; ++i) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
        h_C[i] = 0.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));


    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Copia inicial de datos a la GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 hilosPorBloque(16, 16);
    dim3 bloquesEnGrilla((N + 15) / 16, (N + 15) / 16);

    std::cout << " Calculando multiplicacion de matrices " << N << "x" << N << " en GPU..." << std::endl;

   
    CHECK_CUDA(cudaEventRecord(start));

    // Lanzar Kernel
    multiplicarMatricesGPU<<<bloquesEnGrilla, hilosPorBloque>>>(d_A, d_B, d_C, N);

    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop)); // Esperar a que la GPU termine

   
    float milisegundos = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milisegundos, start, stop));

    // Copiar resultado de vuelta a la CPU
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "[+] Calculo en GPU finalizado." << std::endl;
    std::cout << "    - Tiempo de ejecucion del Kernel (GPU): " << milisegundos << " ms\n" << std::endl;

    std::cout << "[+] Imprimiendo matriz...\n" << std::endl;

    for (int i = 0; i < N; ++i) {
        std::cout << "Fila " << std::setw(3) << i << " [ ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(6) << h_C[i * N + j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\n[+] Impresion de la matriz de " << N << "x" << N << " finalizada con exito." << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}