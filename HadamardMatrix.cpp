#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>

#define peakPerformance 281.28

using namespace std;

// Build Hadamar matrix
int main(int argc, char* argv[]) {
    int** result; // result matrix
    int* residues;
    int k;
    int p;
    int min_n;
    int max_n;
    int n;
    int r; // iteration number per a processor
    int m;
    int r1, r2;
    int r3, r4;
    int i1, j1;
    int rank;
    int s;
    int sum;
    int min_proc;
    int max_proc;
    int step;
    double t_start, t_finish;
    ofstream fout1, fout2, fout3;
    double ex_time;
    int op_number;
    double performance;
    double efficiency;
    int next;
    //
    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    //MPI_Comm_size (MPI_COMM_WORLD, &p);	/* get number of processes */
    if (rank == 0) {
        fout1.open("performance2.txt",std::ios_base::app);
        fout2.open("efficiency2.txt",std::ios_base::app);
        fout3.open("time2.txt",std::ios_base::app);
    }
    min_n = atoi(argv[1]); // min matrix dimension
    max_n = atoi(argv[2]); // max matrix dimension
    min_proc = atoi(argv[3]);
    max_proc = atoi(argv[4]);
    step = atoi(argv[5]);
    p = min_proc;
    while (p <= max_proc) {
    //
        n=min_n;
        while (n <= max_n) {
            next = false;
            if (rank == 0) {
                // Matrix size n must be divisible by 4
                // n-1 must be a prime number
                if ((n % 4) != 0) {
                    next = true;
                } else {
                    m = static_cast<int>(sqrt(static_cast<float>(n-1)));
                    for (int l=2; l<=m; l++) {
                        if (((n-1) % l) == 0) {
                            next = true;
                            break;
                        }
                    }
                }
            }
            MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);
            // int MPI_Barrier(MPI_COMM_WORLD); // synchronization point
            if (next == true) {
                n++;
                continue;
            }
            //cout << n;
            // n meets requirements
            if (rank < p) {
                result = new int*[n];
                for (int i=0; i<n; i++) {
                    result[i] = new int[n];
                }
            }
            MPI_Barrier(MPI_COMM_WORLD); // synchronization point
            if (rank == 0) {
                t_start =  MPI_Wtime();
            }
            MPI_Barrier(MPI_COMM_WORLD); // synchronization point
            k=n-1; // number of residues
            // 1) Считаем вычеты по модулю k=n-1 и создаем вспомогательный массив вычетов residues
            // Достаточно посчитать значения (1^2 mod k), (2^2 mod k), ... , (((k-1)/2)^2 mod k) - эти значениями и будут являться вычетами.
            // Строим массив residues размера 2*k, в котором первые p элементов заполняются по следующей схеме: 1, если индекс элемента - вычет, -1 - иначе.
            // Вторая половина массива получается дублированием первой половины
            if (rank < p) {
                residues = new int[2*k];
                // Заполняем массив
                for (int i=0; i<k; i++) {
                    residues[i] = -1;
                }
                r = (k-1) / 2;
                for (int i=1; i<= r; i++) {
                    residues[i*i % k] = 1;
                }
                r = 2 * k;
                for (int i=k; i<r; i++) {
                    residues[i] = residues[i-k];
                }
                // 2) Построение матрицы Пейли
                r1 = n / p;
                r2 = n % p;
                // Заполнение первой строки 1-цами
                s = (rank+1)*r1;
                for (int j=rank*r1; j<s; j++) {
                    result[0][j] = 1;
                }
                if (rank < r2) {
                    result[0][n-r2+rank] = 1;
                }
                // Заполнение первого столбца 1-цами
                s = (rank+1)*r1;
                for (int i=rank*r1; i<s; i++) {
                    result[i][0] = 1;
                }
                if (rank < r2) {
                    result[n-r2+rank][0] = 1;
                }
                // Заполнение оставшейся части матрицы (она равна матрице Q-I, где Q - матрица Джекобстола) с использованием вспомогательного массива вычетов residues
                r1 = (n-1) / p;
                r2 = (n-1) % p;
                s = 1+ (rank+1)* r1;
                for (int i=1+rank*r1; i<s; i++) {
                    for (int j=1; j<n; j++) {
                        result[i][j] = residues[j-i+k];
                    }
                }
                sum = r2* (n-1);
                r3 = sum / p;
                r4 = sum % p;
                s = (rank+1)*r3;
                for (int i=rank*r3; i<s; i++) {
                    i1 = (i-1) / k + n - r2;
                    j1 = (i-1) % k + 1;
                    result[i1][j1] = residues[j1-i1+k];
                }
                if (rank < r4) {
                    i1 = (sum-r4+rank-1) / k + n - r2;
                    j1 = (sum-r4+rank-1) % k + 1;
                    result[i1][j1] = residues[j1-i1+k];
                }
                // 3) Delete data
                for (int i=0; i<n; i++) {
                    delete result[i];
                }
                delete result;
                delete residues;
            }
            //
            MPI_Barrier(MPI_COMM_WORLD); // synchronization point
            if (rank == 0) {
                t_finish = MPI_Wtime();
                ex_time = t_finish - t_start;
                op_number = n*n + n * 7 / 2 - 7;
                performance = static_cast<double>(op_number) / ex_time / 1000000000.0;
                efficiency = performance * 100 / peakPerformance / p;
            }
            //
            // 4) Print result
            if (rank == 0) {
                fout1 << p << " " << n << " " << performance << endl;
                fout2 << p << " " << n << " " << efficiency << endl;
                fout3 << p << " " << n << " " << ex_time << endl;
            }
            //if (n == 0) {
                //MPI_Send(result[p][p+n],1,MPI_INT,0,0,MPI_COMM_WORLD);
            //}
            //fout1 << p << " " << n << " " << ex_time << endl;
            //
            n++;
        }
        if (rank == 0) {
            fout1 << endl;
            fout2 << endl;
            fout3 << endl;
        }
    p += step;
    //
    }
    if (rank == 0) {
        fout1.close();
        fout2.close();
        fout3.close();
    }
    MPI_Finalize();
}
