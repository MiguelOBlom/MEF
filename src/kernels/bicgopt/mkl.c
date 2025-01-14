#include "util.h"
#include "stdio.h"
#include <mkl.h>

long unsigned int N = trueX * trueX + 2 * trueX + 2 * trueX;

void experiment (float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float *A = &Data[0];
    float *S = &A[trueX * trueX];
    float *Q = &S[trueX];
    float *P = &Q[trueX];
    float *R = &P[trueX];

    cblas_sgemv(CblasRowMajor, CblasTrans, trueX, trueX, 1.0f, A, trueX, R, 1, 0.0f, S, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, trueX, trueX, 1.0f, A, trueX, P, 1, 0.0f, Q, 1);
}