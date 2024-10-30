#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// Estructura de una matriz 
typedef struct {
    int width;      // Número de columnas
    int height;     // Número de filas
    int* elements;  // Elementos de la matriz
} Matrix;


// Kernel para multiplicación de matrices con cache-blocking
template <const int BLOCKSIZE> // Tamaño del bloque con memoria compartida
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const int *A, const int *B,
                                       float beta, int *C) {

    // El bloque de salida que queremos calcular en este bloque de hilos          
    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

    // Asignar buffer para el bloque actual en memoria compartida rápida
    // La memoria compartida es compartida entre todos los hilos en un bloque
    __shared__ int As[BLOCKSIZE][BLOCKSIZE];
    __shared__ int Bs[BLOCKSIZE][BLOCKSIZE];

    // Inicializar el acumulador temporal para este bloque a 0
    int tmp = 0;
    for (int bkIdx = 0; bkIdx < (K + BLOCKSIZE - 1) / BLOCKSIZE; ++bkIdx) {
        // Hacer que cada hilo cargue uno de los elementos en A y B
        // Hacer que threadCol (=threadIdx.x) sea el índice consecutivo
        // para permitir el acceso colectivo a la memoria global
        if (row < M && bkIdx * BLOCKSIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + bkIdx * BLOCKSIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < N && bkIdx * BLOCKSIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(bkIdx * BLOCKSIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        // Bloquear hilos del bloque hasta que la memoria compartida este llena
        __syncthreads();

        // Aplicar el producto punto entre la fila de A y la columna de B
        // en el bloque actual de memoria compartida
        for (int e = 0; e < BLOCKSIZE; ++e)
            tmp += As[threadIdx.y][e] * Bs[e][threadIdx.x];

        // Sincronizar nuevamente al final para evitar que los hilos más rápidos
        // carguen el siguiente bloque en la caché antes de que los hilos más lentos hayan terminado
        __syncthreads();
    }

    // Escribir el resultado en la matriz de salida
    if (row < M && col < N)
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}

#define BLOCKSIZE 32    // Tamaño del bloque con memoria compartida

// Función para inicializar matrices en el dispositivo y ejecutar el kernel
void matrixMultiplyCUDA(const Matrix &A, const Matrix &B, Matrix &C) {

    // Inicializar matrices en el dispositivo
    int *d_A, *d_B, *d_C;

    // Reservar memoria en el dispositivo para las matrices
    // y copiar los datos de las matrices del host al dispositivo
    size_t size = A.width * A.height * sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A.elements, size, cudaMemcpyHostToDevice);

    size = B.width * B.height * sizeof(int);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B.elements, size, cudaMemcpyHostToDevice);

    size = C.width * C.height * sizeof(int);
    cudaMalloc(&d_C, size);
    cudaMemset(d_C, 0, size);

    // Configurar la cuadrícula y los bloques de hilos
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid((C.width + BLOCKSIZE - 1) / BLOCKSIZE, (C.height + BLOCKSIZE - 1) / BLOCKSIZE);

    // Iniciar temporizador para medir el tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);    // Crear eventos para medir el tiempo
    cudaEventCreate(&stop);     // de ejecución en CUDA

    // Llamar al kernel para multiplicar matrices con cache-blocking
    // y medir el tiempo de ejecución
    cudaEventRecord(start);     
    sgemm_shared_mem_block<BLOCKSIZE><<<dimGrid, dimBlock>>>(A.height, B.width, A.width, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaEventRecord(stop);

    // Sincronizar el dispositivo y calcular el tiempo de ejecución
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    INFO: Tiempo de ejecucion en CUDA: %f ms\n\n", milliseconds);

    // Copiar el resultado de la matriz de salida del dispositivo al host
    cudaMemcpy(C.elements, d_C, size, cudaMemcpyDeviceToHost);


    // Liberar memoria en el dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destruir eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Función secuencial de multiplicación de matrices
void matrixMultiplySeq(int* A, int* B, int* C, int N) {
    // Iterar sobre las filas de la matriz A
    for (int i = 0; i < N; ++i) {
        // Iterar sobre las columnas de la matriz B
        for (int j = 0; j < N; ++j) {
            int sum = 0;    // Inicializar el acumulador temporal a 0

            // Obtener el producto punto de la fila de A y la columna de B
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            // Asignar el resultado a la matriz de salida
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
        printf("\n");
    }
}

int main() {
    int N = 1024;   // Tamaño de las matrices (N x N)

    printf("--- INICIALIZACION DE MATRICES ---\n");
    // Inicializar matrices
    Matrix A, B, C_seq, C_cuda;

    // Asignar tamaño a las matrices
    printf("    -> Asignacion de dimensiones...\n");
    A.width = B.width = C_seq.width = C_cuda.width = N;
    A.height = B.height = C_seq.height = C_cuda.height = N;

    // Reservar memoria para los elementos de las matrices en el host
    printf("    -> Reservando memoria para los elementos...\n");
    A.elements = (int*)malloc(N * N * sizeof(int));
    B.elements = (int*)malloc(N * N * sizeof(int));
    C_seq.elements = (int*)malloc(N * N * sizeof(int));
    C_cuda.elements = (int*)malloc(N * N * sizeof(int));

    // Inicializar elementos de las matrices con valores aleatorios entre 0 y 9
    printf("    -> Asignacion de valores aleatorios...\n");
    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        A.elements[i] = rand() % 10;
        B.elements[i] = rand() % 10;
    }

    printf("    EXITO: Matrices inicializadas\n\n");



    printf("--- MULTIPLICACION DE MATRICES ---\n");

    // Multiplicar matrices secuencialmente
    printf("   SECUENCIAL\n");
    printf("    -> Multiplicando matrices de %d x %d ...\n", N, N);
    auto start = std::chrono::high_resolution_clock::now(); // Iniciar temporizador
    matrixMultiplySeq(A.elements, B.elements, C_seq.elements, N);
    auto end = std::chrono::high_resolution_clock::now();   // Finalizar temporizador
    // Calcular tiempo de ejecución secuencial
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    printf("    INFO: Tiempo de ejecucion secuencial: %f ms\n\n", duration_ms.count());
    
    // Multiplicar matrices con CUDA
    printf("   CUDA\n");
    printf("    -> Multiplicando matrices de %d x %d ...\n", N, N);
    matrixMultiplyCUDA(A, B, C_cuda);

    // Verificar si las matrices coinciden
    printf("--- VERIFICACION DE RESULTADOS ---\n");
    bool match = true;
    for (int i = 0; i < N * N; i++) {
        if (C_seq.elements[i] != C_cuda.elements[i]) {
            match = false;
            break;
        }
    }
    printf("    Las matrices %s\n\n", match ? "coinciden" : "no coinciden");

    // Descomentar para imprimir matrices
    // printf("Matriz A:\n");
    // printMatrix(&A);
    // printf("\nMatriz B:\n");
    // printMatrix(&B);
    // printf("\nMatriz C (secuencial):\n");
    // printMatrix(&C_seq);
    // printf("\nMatriz C (CUDA):\n");
    // printMatrix(&C_cuda);


    // Liberar memoria
    free(A.elements);
    free(B.elements);
    free(C_seq.elements);
    free(C_cuda.elements);

    return 0;
}
