#include "util.h"
#include <stdio.h>
#include "mkl.h"

long unsigned int N = 2 * trueP + trueP * trueP;

void experiment(float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float Alpha = 4.5f;

    float * A = (float *) &Data[0];
    float * W = (float *) &A[trueP * trueP];
    float * X = (float *) &W[trueP];
    
    // for i in range(trueP):
    //     for j in range(trueP):
    //         w[i] = w[i] + 4.5 * a[i][j] * x[j]
    cblas_sgemv(CblasRowMajor, CblasNoTrans, trueP, trueP, Alpha, A, trueP, X, 1, 1.0f, W, 1);
}
