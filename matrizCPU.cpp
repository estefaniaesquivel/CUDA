#include <iostream>
#include <iomanip> 
#include <vector>
#include <chrono> 

// cl /O2 /EHsc matrizCPU.cpp
// .\matrizCPU.exe > matrizCPU.txt
//
int main() {
    int N = 3000; 
    int totalElementos = N * N;

    std::vector<float> h_A(totalElementos, 2.0f);
    std::vector<float> h_B(totalElementos, 3.0f);
    std::vector<float> h_C(totalElementos, 0.0f);

    std::cout << "Calculando multiplicacion de matrices " << N << "x" << N << " en CPU..." << std::endl;

    auto inicioTotal = std::chrono::high_resolution_clock::now();

    auto inicioCalculo = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float suma = 0.0f;
            for (int k = 0; k < N; ++k) {
                suma += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = suma;
        }
    }

    auto finCalculo = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempoCalculoMS = finCalculo - inicioCalculo;
    std::chrono::duration<double> tiempoCalculoSeg = finCalculo - inicioCalculo;

    std::cout << "Calculo en CPU finalizado." << std::endl;
    std::cout << "    - Tiempo exclusivo de computo (CPU): " << tiempoCalculoSeg.count() << " s (" << tiempoCalculoMS.count() << " ms)\n" << std::endl;

    std::cout << "Imprimiendo matriz...\n" << std::endl;

    for (int i = 0; i < N; ++i) {
        std::cout << "Fila " << std::setw(3) << i << " [ ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(6) << h_C[i * N + j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    auto finTotal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempoTotal = finTotal - inicioTotal;

    std::cout << "\nImpresion de la matriz de " << N << "x" << N << " finalizada con exito." << std::endl;
    std::cout << "TIEMPO TOTAL (Calculo + Impresion): " << tiempoTotal.count() << " segundos." << std::endl;

    return 0;
}