#include <iostream>
#include <iomanip> // Para std::setw
#include <vector>
#include <chrono>  // Para medir tiempo de ejecución

//cl /O2 matriz_cpu.cpp
//.\matriz_cpu.exe > matriz_cpu.txt

int main() {
    int N = 1000; 
    int totalElementos = N * N;

    // 1. Reservar memoria en CPU (Host)
    std::vector<float> h_A(totalElementos, 2.0f);
    std::vector<float> h_B(totalElementos, 3.0f);
    std::vector<float> h_C(totalElementos, 0.0f);

    std::cout << "Calculando multiplicacion de matrices " << N << "x" << N << " en CPU..." << std::endl;

    // 2. Medir tiempo de cálculo en CPU
    auto inicio = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float suma = 0.0f;
            for (int k = 0; k < N; ++k) {
                suma += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = suma;
        }
    }

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempoCPU = fin - inicio;

    std::cout << "Calculo en CPU finalizado." << std::endl;
    std::cout << "    - Tiempo de computo (CPU): " << tiempoCPU.count() << " ms\n" << std::endl;

    std::cout << "Imprimiendo matriz...\n" << std::endl;

    // 3. IMPRESIÓN FORMATADA: Exactamente igual al formato CUDA
    for (int i = 0; i < N; ++i) {
        std::cout << "Fila " << std::setw(3) << i << " [ ";
        
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(6) << h_C[i * N + j] << " ";
        }
        
        std::cout << "]" << std::endl;
    }

    std::cout << "\n[+] Impresion de la matriz de " << N << "x" << N << " finalizada con exito." << std::endl;

    return 0;
}