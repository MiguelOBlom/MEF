#include "util.h"
#include "stdio.h"
#include <cblas.h>

long unsigned int N = trueX * trueX + 2 * trueX + 2 * trueX;

void experiment (float * Data) {
    float *A = &Data[0];
    float *S = &A[trueX * trueX];
    float *Q = &S[trueX];
    float *P = &Q[trueX];
    float *R = &P[trueX];

    float zero = 0.0f;

    cblas_sgemv(CblasRowMajor, CblasTrans, trueX, trueX, 1.0f, A, trueX, R, 1, zero, S, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, trueX, trueX, 1.0f, A, trueX, P, 1, zero, Q, 1);
}