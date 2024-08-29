#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int MAX_NUMBERS = 2000;
    float numbers[MAX_NUMBERS];
    float result_sum = 0.0;
    int random_amount;

    int M = 5 * (world_size - 1); // M vezes o número de escravos

    if (world_rank == 0)
    {
        srand(time(NULL));

        for (int m = 0; m < M; m++)
        {
            // Gera um número aleatório entre 1000 e 2000
            int min = 1000;
            int max = 2000;
            random_amount = min + rand() % (max - min + 1);

            for (int i = 0; i < random_amount; i++)
            {
                // Gera numero aleatorio de 0.0 a 99.0 para cada posicao de "numbers"
                numbers[i] = ((float)rand() / RAND_MAX) * 99.0;
            }

            // Envia os números gerados para o escravo que estiver livre
            int slave_rank;
            MPI_Recv(&slave_rank, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            MPI_Send(&random_amount, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
            MPI_Send(numbers, random_amount, MPI_FLOAT, slave_rank, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < world_size; i++)
        {
            float value;
            MPI_Recv(&value, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result_sum += value;
            printf("Received %.2f from process %d\n", value, i);
        }

        printf("Total sum from all slaves: %.2f\n", result_sum);
    }
    else
    {
        for (int m = 0; m < M / (world_size - 1); m++)
        {
            // Notifica o mestre que o escravo está pronto para receber dados
            MPI_Send(&world_rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

            // Recebe a quantidade de números e os números em si
            MPI_Recv(&random_amount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(numbers, random_amount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Calcula a soma dos números recebidos
            result_sum = 0.0;
            for (int i = 0; i < random_amount; i++)
            {
                result_sum += numbers[i];
            }

            // Envia a soma de volta para o mestre
            MPI_Send(&result_sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

            printf("Process %d sent sum %.2f to Master\n", world_rank, result_sum);
        }
    }

    MPI_Finalize();
    return 0;
}