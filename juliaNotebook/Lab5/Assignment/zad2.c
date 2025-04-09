#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_bessel.h>
#include <sys/times.h>
#include <unistd.h>
#include <fcntl.h>
struct tms tms_start, tms_end;
clock_t clock_start, clock_end;

void start_time(){
    clock_start = times(&tms_start);
}

void clear_matrix(double **M, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            M[i][j] = 0.0;
}

double end_time(){
    clock_end = times(&tms_end);
    int tics = sysconf(_SC_CLK_TCK);
    double real = (double)(clock_end-clock_start) / tics;
    return real;
}
void naive_multiplication(double** A, double** B, double** C, int size){
    for (int i = 0; i < size;i++){
        for (int j = 0; j < size; j++){
            for (int k = 0; k < size; k ++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }     
    }
}
void better_multiplication(double** A, double** B, double** C, int size){
    for (int i = 0; i < size; i++){
        for (int k = 0; k < size; k ++){
            for (int j = 0; j < size; j++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }    
}

void blas_multiplication(double* a, double* b, double* c, int rows){
    gsl_matrix_view D = gsl_matrix_view_array(a, rows, rows);
    gsl_matrix_view E = gsl_matrix_view_array(b, rows, rows);
    gsl_matrix_view F = gsl_matrix_view_array(c, rows, rows);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                  1.0, &D.matrix, &E.matrix,
                  0.0, &F.matrix);
}

int main(int argc, char* argv[]){
    double **A, **B, **C;
    double *a, *b, *c;
    double time1, time2, time3;
    FILE *report = fopen("resultsO4.csv","a");
    fprintf(report, "size,sample,time_naive,time_better,time_blas\n");
    for (int i = 10; i <= 760; i += 10){
        A = calloc(i,sizeof(double *));
        B = calloc(i,sizeof(double *));
        C = calloc(i,sizeof(double *));
        a = calloc(i*i, sizeof(double));
        b = calloc(i*i, sizeof(double));
        c = calloc(i*i, sizeof(double));
        for (int j = 0; j < i; j++){
            A[j] = calloc(i,sizeof(double));
            B[j] = calloc(i,sizeof(double));
            C[j] = calloc(i,sizeof(double));
        }
        for (int j = 0; j < 10; j++){
            for (int k = 0; k < i; k ++){
                for (int l = 0; l < i; l ++){
                    A[k][l] = (double)(rand() % 1000) / 100.0;
                    B[k][l] = (double)(rand() % 1000) / 100.0;
                }
            }
            start_time();
            naive_multiplication(A,B,C,i);
            time1 = end_time();

            clear_matrix(C, i);

            start_time();
            better_multiplication(A,B,C,i);
            time2 = end_time();

            clear_matrix(C, i);

            for (int k = 0; k < i*i; k++){
                a[k] = A[k/i][k%i];
                b[k] = B[k/i][k%i];
            }   
            start_time();
            blas_multiplication(a,b,c,i);
            time3 = end_time();

            clear_matrix(C, i);

            fprintf(report,"%d,%d,%f,%f,%f\n",i,j,time1,time2,time3);
        }
        for (int j = 0; j < i; j++){
            free(A[j]);
            free(B[j]);
            free(C[j]);
        }
        free(A);
        free(B);
        free(C);
        free(a);
        free(b);
        free(c);
        printf("%d\n", i);
    }
    fclose(report);
    return 0;
}