#include "util.h"
#include <stdio.h>
#include "mkl.h"

long unsigned int N = 2 * trueP;

void experiment(float * Data) {
    mkl_enable_instructions(MKL_ENABLE_AVX2);
    float * X = (float *) &Data[0];
    float * Z = (float *) &X[trueP];

    // for i in range(trueP):
    //     x[i] += z[i]
    cblas_saxpy(trueP, 1.0f, Z, 1, X, 1);
}
