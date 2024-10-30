#include <stdio.h>
#include <chrono>
#include <cstdlib>

// Estructura de una matriz 
typedef struct {
    int width;
    int height;
    int* elements;
} Matrix;

// Función secuencial de multiplicación de matrices
void matrixMultiplySeq(int* A, int* B, int* C, int N) {
    // Iterar sobre las filas de A
    for (int i = 0; i < N; ++i) {
        // Iterar sobre las columnas de B
        for (int j = 0; j < N; ++j) {
            
            int sum = 0; // Inicializar suma en 0

            // Obtener el producto punto de la fila i de A y la columna j de B
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }

            // Almacenar el resultado en C[i, j]
            C[i * N + j] = sum;
        }
    }
}

// Función para imprimir una matriz
void printMatrix(const Matrix* mat) {
    for (int i = 0; i < mat->height; ++i) {
        for (int j = 0; j < mat->width; ++j) {
            printf("%d ", mat->elements[i * mat->width + j]);
        }
        printf("\n"); // Salto de línea al final de cada fila
    }
}

int main() {
    // Dimensiones de las matrices cuadradas
    int N = 5;

    // Inicializar matrices en el host (CPU)
    Matrix A, B, C_seq;

    A.width = B.width = C_seq.width = N;
    A.height = B.height = C_seq.height = N;

    A.elements = (int*)malloc(N * N * sizeof(int));
    B.elements = (int*)malloc(N * N * sizeof(int));
    C_seq.elements = (int*)malloc(N * N * sizeof(int));

    // Rellenar A y B con valores aleatorios
    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        A.elements[i] = rand() % 10;  // Números más pequeños para facilitar la visualización
        B.elements[i] = rand() % 10;
    }

    
    // Multiplicación de matrices secuencial
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplySeq(A.elements, B.elements, C_seq.elements, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    printf("Tiempo secuencial: %f ms\n", duration_ms.count());
    

    // Imprimir matrices
    printf("Matriz A:\n");
    printMatrix(&A);

    printf("\nMatriz B:\n");
    printMatrix(&B);

    printf("\nMatriz C (secuencial):\n");
    printMatrix(&C_seq);

    // Liberar memoria
    free(A.elements);
    free(B.elements);
    free(C_seq.elements);

    return 0;
}
