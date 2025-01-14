#include "util.h"
#include <stdio.h>
#include "mkl.h"

long unsigned int N = 4 * trueP + trueP * trueP;

void experiment(float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float * A = (float *) &Data[0];
    float * U1 = (float *) &A[trueP * trueP];
    float * V1 = (float *) &U1[trueP];
    float * U2 = (float *) &V1[trueP];
    float * V2 = (float *) &U2[trueP];

    // for i in range(trueP):
    //     for j in range(trueP):
    //         a[i][j] += u1[i] * v1[j] + u2[i] * v2[j]
    cblas_sger(CblasRowMajor, trueP, trueP, 1.0f, U1, 1, V1, 1, A, trueP);
    cblas_sger(CblasRowMajor, trueP, trueP, 1.0f, U2, 1, V2, 1, A, trueP);
}
