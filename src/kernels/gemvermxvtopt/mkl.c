#include "util.h"
#include <stdio.h>
#include "mkl.h"

long unsigned int N = 2 * trueP + trueP * trueP;

void experiment(float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float Beta = -1.1f;

    float * A = (float *) &Data[0];
    float * X = (float *) &A[trueP * trueP];
    float * Y = (float *) &X[trueP];

    // for i in range(trueP):
    //     for j in range(trueP):
    //         x[i] = x[i] + -1.1 * a[j][i] * y[j]
    cblas_sgemv(CblasRowMajor, CblasTrans, trueP, trueP, Beta, A, trueP, Y, 1, 1.0f, X, 1);
}
