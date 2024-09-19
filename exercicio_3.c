/**
 * Otávio Augusto Souza Martins 2022.1.08.016
 * Multiplicação de matrizes
 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

// Função para alocar uma matriz linear
int *allocate_matrix(int rows, int cols) {
    return (int *)malloc(rows * cols * sizeof(int));
}

// Função para imprimir uma matriz linear
void print_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    int rows_A, cols_A, rows_B, cols_B;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 3) {
        if (rank == MASTER) {
            printf("Uso: %s <linhas_A> <colunas_B>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Inicializa as dimensões das matrizes
    rows_A = atoi(argv[1]);
    cols_B = atoi(argv[2]);
    cols_A = num_procs;  // Cada processo tratará uma coluna de A
    rows_B = num_procs;  // O número de linhas de B também é igual ao número de processos

    // Declaração das matrizes A, B e C
    int *A = NULL, *B = NULL, *C = NULL;

    if (rank == MASTER) {
        // Aloca as matrizes A, B e C no processo mestre
        A = allocate_matrix(rows_A, cols_A);
        B = allocate_matrix(rows_B, cols_B);
        C = allocate_matrix(rows_A, cols_B);

        // Inicializa a matriz A com valores aleatórios
        printf("Matriz A:\n");
        srand(time(NULL));  // Gera uma semente de números aleatórios
        for (int i = 0; i < rows_A * cols_A; i++) {
            A[i] = rand() % 10;
        }
        print_matrix(A, rows_A, cols_A);

        // Inicializa a matriz B com valores aleatórios
        printf("Matriz B:\n");
        for (int i = 0; i < rows_B * cols_B; i++) {
            B[i] = rand() % 10;
        }
        print_matrix(B, rows_B, cols_B);
    }

    // Buffer para armazenar uma linha de A e uma linha do resultado local
    int *row_A = (int *)malloc(cols_A * sizeof(int));
    int *local_C = (int *)malloc(cols_B * sizeof(int));

    // Distribui as linhas de A entre os processos
    MPI_Scatter(A, cols_A, MPI_INT, row_A, cols_A, MPI_INT, MASTER, MPI_COMM_WORLD);

    // A matriz B precisa ser enviada para todos os processos
    if (rank != MASTER) {
        B = allocate_matrix(rows_B, cols_B);  // Aloca B para processos não mestres
    }
    MPI_Bcast(B, rows_B * cols_B, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calcula a multiplicação da linha de A recebida pela matriz B
    for (int i = 0; i < cols_B; i++) {
        local_C[i] = 0;
        for (int j = 0; j < cols_A; j++) {
            local_C[i] += row_A[j] * B[j * cols_B + i];
        }
    }

    // Coleta os resultados de todos os processos na matriz C no processo mestre
    MPI_Gather(local_C, cols_B, MPI_INT, C, cols_B, MPI_INT, MASTER, MPI_COMM_WORLD);

    // O processo mestre imprime o resultado final
    if (rank == MASTER) {
        printf("Matriz C (Resultado da multiplicação):\n");
        print_matrix(C, rows_A, cols_B);
    }

    // Libera a memória alocada
    free(row_A);
    free(local_C);
    if (rank == MASTER) {
        free(A);
        free(B);
        free(C);
    } else {
        free(B);
    }

    MPI_Finalize();
    return 0;
}
