#include "util.h"
#include <stdio.h>
#include "mkl.h"

const float alpha = 4.5f;
const float beta = -1.1f;

long unsigned int N = trueP * (trueP + 2);

void experiment (float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float *A = (float *) &Data[0];
    float *B = (float *) &A[trueP * trueP];
    float *C = (float *) &B[trueP];
    
    cblas_sgemv (CblasRowMajor, CblasNoTrans, trueP, trueP, alpha, A, trueP, B, 1, beta, C, 1);
}
