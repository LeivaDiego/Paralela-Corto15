#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// Estructura de una matriz 
typedef struct {
    int width;
    int height;
    int* elements;
} Matrix;


// Kernel para multiplicación de matrices con cache-blocking
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const int *A, const int *B,
                                       float beta, int *C) {
    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
    __shared__ int As[BLOCKSIZE][BLOCKSIZE];
    __shared__ int Bs[BLOCKSIZE][BLOCKSIZE];

    int tmp = 0;
    for (int bkIdx = 0; bkIdx < (K + BLOCKSIZE - 1) / BLOCKSIZE; ++bkIdx) {
        if (row < M && bkIdx * BLOCKSIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + bkIdx * BLOCKSIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < N && bkIdx * BLOCKSIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(bkIdx * BLOCKSIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int e = 0; e < BLOCKSIZE; ++e)
            tmp += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}

#define BLOCKSIZE 32

// Función para inicializar matrices en el dispositivo y ejecutar el kernel
void matrixMultiplyCUDA(const Matrix &A, const Matrix &B, Matrix &C) {
    int *d_A, *d_B, *d_C;
    size_t size = A.width * A.height * sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A.elements, size, cudaMemcpyHostToDevice);

    size = B.width * B.height * sizeof(int);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B.elements, size, cudaMemcpyHostToDevice);

    size = C.width * C.height * sizeof(int);
    cudaMalloc(&d_C, size);
    cudaMemset(d_C, 0, size);

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid((C.width + BLOCKSIZE - 1) / BLOCKSIZE, (C.height + BLOCKSIZE - 1) / BLOCKSIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_shared_mem_block<BLOCKSIZE><<<dimGrid, dimBlock>>>(A.height, B.width, A.width, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecucion en CUDA: %f ms\n", milliseconds);

    cudaMemcpy(C.elements, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Función secuencial de multiplicación de matrices
void matrixMultiplySeq(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
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
    int N = 1024;

    Matrix A, B, C_seq, C_cuda;

    A.width = B.width = C_seq.width = C_cuda.width = N;
    A.height = B.height = C_seq.height = C_cuda.height = N;

    A.elements = (int*)malloc(N * N * sizeof(int));
    B.elements = (int*)malloc(N * N * sizeof(int));
    C_seq.elements = (int*)malloc(N * N * sizeof(int));
    C_cuda.elements = (int*)malloc(N * N * sizeof(int));

    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        A.elements[i] = rand() % 10;
        B.elements[i] = rand() % 10;
    }

    printf("SECUENCIAL -> Multiplicando matrices de %d x %d ...\n", N, N);
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplySeq(A.elements, B.elements, C_seq.elements, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    printf("Tiempo de ejecucion secuencial: %f ms\n\n", duration_ms.count());
    
    printf("CUDA -> Multiplicando matrices de %d x %d ...\n", N, N);
    matrixMultiplyCUDA(A, B, C_cuda);

    bool match = true;
    for (int i = 0; i < N * N; i++) {
        if (C_seq.elements[i] != C_cuda.elements[i]) {
            match = false;
            break;
        }
    }
    printf("Las matrices %s\n", match ? "coinciden" : "no coinciden");

    // printf("Matriz A:\n");
    // printMatrix(&A);
    // printf("\nMatriz B:\n");
    // printMatrix(&B);
    // printf("\nMatriz C (secuencial):\n");
    // printMatrix(&C_seq);
    // printf("\nMatriz C (CUDA):\n");
    // printMatrix(&C_cuda);

    free(A.elements);
    free(B.elements);
    free(C_seq.elements);
    free(C_cuda.elements);

    return 0;
}
