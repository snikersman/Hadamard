#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>

#define peakPerformance 281.28

using namespace std;

// n=4k, n-1 - a prime number
// Build Hadamar matrix

void split1(int n, int numOfProc, int & a1, int & a2)
{
    int rank;
    int k1, k2;
    //
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    //
    numOfProc--;
    rank--;
    k1 = n / numOfProc;
    k2 = n % numOfProc;
    if (rank < k2) {
        a1 = rank * (k1 + 1);
        a2 = a1 + k1;
    } else {
        a1 = n - k1 * (numOfProc - rank);
        a2 = a1 + k1 - 1;
    }
}

void split2(int n, int numOfProc, int & a1, int & a2)
{
    int rank;
    int k1, k2;
    int n_2;
    //
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    //
    numOfProc--;
    rank--;
    n_2 = n * n;
    k1 = n_2 / numOfProc;
    k2 = n_2 % numOfProc;
    if (rank < k2) {
        a1 = rank * (k1 + 1);
        a2 = a1 + k1;
    } else {
        a1 = n_2 - k1 * (numOfProc - rank);
        a2 = a1 + k1 - 1;
    }
}

void buildHadamardMatrix(int n, int numOfProc, vector< vector<int> > & result, MPI_Comm comm)
{
    int a1, a2;
    int rank;
    int p;
    int buf1[2]; // ind val
    int buf2[3]; // ind1 ind2 val
    int* residues;
    int s;
    MPI_Status status;
    // --------------------------------------
    MPI_Comm_rank (comm, &rank);	/* get current process id */
    // -------------------------------
    if (rank == 0) {
        // 1) matrix size changing
        result.resize(n);
        for (int i=0; i<n; i++) {
            result[i].resize(n);
        }
    }
    // ------------------------------
    // 2) residue array building
    p=n-1;
    residues = new int[2*p];
    // «аполн€ем первую половину массива -1-ми
    if (rank == 0) {
        for (int i=0; i<p; i++) {
            residues[i] = -1;
        }
    }
    /*if (rank == 0) {
        for (int i=0; i<p; i++) {
            MPI_Recv(buf1, 2, MPI_INT, MPI_ANY_SOURCE, 1, comm, &status);
            residues[buf1[0]] = buf1[1];
        }
        //MPI_Bcast(residues, p, MPI_INT, 0, comm);
    } else {
        split1(p, numOfProc, a1, a2);
        for (int i=a1; i<=a2; i++) {
            buf1[0] = i;
            buf1[1] = -1;
            MPI_Send(buf1, 2, MPI_INT, 0, 1, comm);
        }
        //MPI_Bcast(residues, p, MPI_INT, 0, comm);
    }*/
    //
    if (rank == 0) {
        s = (p-1)/2;
        for (int i=1; i<=s; i++) {
            MPI_Recv(buf1, 2, MPI_INT, MPI_ANY_SOURCE, 2, comm, &status);
            residues[buf1[0]] = buf1[1];
        }
        MPI_Bcast(residues, p, MPI_INT, 0, comm);
    } else {
        split1((p-1)/2, numOfProc, a1, a2);
        a1++;
        a2++;
        for (int i=a1; i<=a2; i++) {
            buf1[0] = i*i % p;
            buf1[1] = 1;
            MPI_Send(buf1, 2, MPI_INT, 0, 2, comm);
        }
        MPI_Bcast(residues, p, MPI_INT, 0, comm);
    }
    if (rank == 0) {
        for (int i=0; i<p; i++) {
            MPI_Recv(buf1, 2, MPI_INT, MPI_ANY_SOURCE, 3, comm, &status);
            residues[buf1[0]] = buf1[1];
        }
        MPI_Bcast(&(residues[p]), p, MPI_INT, 0, comm);
    } else {
        split1(p, numOfProc, a1, a2);
        a1 += p;
        a2 += p;
        for (int i=a1; i<=a2; i++) {
            buf1[0] = i;
            buf1[1] = residues[i-p];
            MPI_Send(buf1, 2, MPI_INT, 0, 3, comm);
        }
        MPI_Bcast(&(residues[p]), p, MPI_INT, 0, comm);
    }
    // 3) Paley matrix buildin
    if (rank == 0) {
        // first line filling
        for (int j=0; j<n; j++) {
            result[0][j] = 1;
        }
        // first column filling
        for (int i=1; i<n; i++) {
            result[i][0] = 1;
        }
    }
    // others
    if (rank == 0) {
        s = (n-1) * (n-1);
        for (int i=0; i<s; i++) {
            MPI_Recv(buf2, 3, MPI_INT, MPI_ANY_SOURCE, 4, comm, &status);
            result[buf2[0]][buf2[1]] = buf2[2];
        }
    } else {
        split2(n-1, numOfProc, a1, a2);
        for (int i=a1; i<=a2; i++) {
            buf2[0] = i / (n-1) + 1;
            buf2[1] = i % (n-1) + 1;
            buf2[2] = residues[buf2[1]-buf2[0]+p];
            MPI_Send(buf2, 3, MPI_INT, 0, 4, comm);
        }
    }
    delete residues;
    // -----------------------------------------
}

bool test(int n, vector< vector<int> > & result)
{
    bool flag;
    int s;
    //
    flag = true;
    for (int i1=0; i1<n-1; i1++) {
        s = 0;
        for (int i2=i1+1; i2<n; i2++) {
            for (int j=0; j<n; j++) {
                s += result[i1][j]*result[i2][j];
            }
        }
        if (s != 0) {
            flag = false;
            break;
        }
    }
    for (int j1=0; j1<n-1; j1++) {
        if (flag == false) {
            break;
        }
        s = 0;
        for (int j2=j1+1; j2<n; j2++) {
            for (int i=0; i<n; i++) {
                s += result[i][j1]*result[i][j2];
            }
        }
        if (s != 0) {
            flag = false;
            break;
        }
    }
    return flag;
}

bool printMatrix(int n, vector< vector<int> > & result)
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (result[i][j] == 1) {
                cout << 1 << " ";
            } else {
                cout << "- ";
            }
        }
        cout << "\n";
    }
    cout << "\n";
}

int main(int argc, char* argv[]) {
    int min_n; // min matrix size
    int max_n; // max matrix size
    int min_proc;
    int max_proc;
    int step;
    double t_start, t_finish;
    ofstream fout1, fout2, fout3;
    double ex_time;
    int op_number;
    double performance;
    double efficiency;
    int n; // matrix size
    bool next;
    int m;
    int bt, bp;
    int rank;
    vector< vector<int> > result; // hadamar matrix
    vector<int> residues;
    int numOfProc;
    int color;
    bool erFlag;
    MPI_Comm comm;
    //
    min_proc = atoi(argv[1]);
    max_proc = atoi(argv[2]);
    step = atoi(argv[3]);
    min_n = atoi(argv[4]); // min matrix dimension
    max_n = atoi(argv[5]); // max matrix dimension
    bt = atoi(argv[6]); // flag of testing (if bt==0 => without testing)
    bp = atoi(argv[7]); // flag of printing (if bp==0 => without printing)
    // -------------------------------------------------------------------
    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
    // MPI_Comm_size(MPI_COMM_WORLD, &numOfProc); // get number of processors
    // -------------------------------------------------------------------
    erFlag = false;
    if (rank == 0) {
        fout1.open("performance.txt"); //,std::ios_base::app);
        fout2.open("efficiency.txt"); //,std::ios_base::app);
        fout3.open("time.txt"); //,std::ios_base::app);
    }
    //
    numOfProc = min_proc;
    while (numOfProc <= max_proc) {
        if (rank < numOfProc) {
            color = 0;
        } else {
            color = 1;
        }
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm);
        if (color == 0) {
            n = min_n;
            while (n <= max_n) {
                // --------------------------------------------------------
                if (rank == 0) {
                    next = false;
                    // Matrix size n must be divisible by 4
                    // n-1 must be a prime number
                    if ((n % 4) != 0) {
                        next = true;
                    }
                    if (next == false) {
                        m = static_cast<int>(sqrt(static_cast<float>(n-1)));
                        for (int l=2; l<=m; l++) {
                            if (((n-1) % l) == 0) {
                                next = true;
                                break;
                            }
                        }
                    }
                }
                MPI_Barrier(comm); // synchronization point
                MPI_Bcast(&next, 1, MPI_INT, 0, comm);
                //MPI_Barrier(comm); // synchronization point
                if (next == true) {
                    n++;
                    continue;
                }
                // -----------------------------------------------------------
                // n meets requirements
                //
                MPI_Barrier(comm); // synchronization point
                if (rank == 0) {
                    t_start =  MPI_Wtime();
                }
                buildHadamardMatrix(n, numOfProc, result, comm);
                if (rank == 0) {
                    t_finish =  MPI_Wtime();
                    ex_time = t_finish - t_start;
                    op_number = n*n + n * 7 / 2 - 7;
                    performance = static_cast<double>(op_number) / ex_time / 1000000000.0;
                    efficiency = performance * 100 / peakPerformance / numOfProc;
                }
                // MPI_Barrier(comm); // synchronization point
                // --------------------------------------------------------
                if (rank == 0) {
                    if (bt != 0) {
                        // need to test;
                        if (test(n, result) == true) {
                            ;
                            // cout << "You are perfect!!!:) (" << n << ")\n";
                        } else {
                            erFlag = true;
                            // cout << "You are wrong! (" << n << ")\n";
                        }
                    }
                    if (bp != 0) {
                        // need to print;
                        printMatrix(n,result);
                    }
                    // Print result
                    fout1 << numOfProc << " " << n << " " << performance << endl;
                    fout2 << numOfProc << " " << n << " " << efficiency << endl;
                    fout3 << numOfProc << " " << n << " " << ex_time << endl;
                }
                // ----------------------------------------------------------
                n++;
            }
        }
        MPI_Comm_free(&comm);
        numOfProc += step;
    }
    if (rank == 0) {
        fout1.close();
        fout2.close();
        fout3.close();
    }
    if (rank == 0) {
        if (bt != 0) {
            if (erFlag == true) {
                cout << "You are wrong!\n";
            } else {
                cout << "You are perfect!!!\n";
            }
        }
    }
    MPI_Finalize();
}
