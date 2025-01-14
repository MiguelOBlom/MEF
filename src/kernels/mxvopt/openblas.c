#include "util.h"
#include <stdio.h>
#include "cblas.h"

const float alpha = 4.5f;
const float beta = -1.1f;

long unsigned int N = trueP * (trueP + 2);

void experiment (float * Data) {
    float *A = (float *) &Data[0];
    float *B = (float *) &A[trueP * trueP];
    float *C = (float *) &B[trueP];
    
    cblas_sgemv (CblasRowMajor, CblasNoTrans, trueP, trueP, alpha, A, trueP, B, 1, beta, C, 1);
}
