#include "util.h"
#include <stdio.h>
#include "cblas.h"

long unsigned int N = 8 * trueP + trueP * trueP;

float alpha = 4.5f;
float beta = -1.1f;

void experiment(float * Data) {
    float * A = (float *) &Data[0];
    float * U1 = (float *) &A[trueP * trueP];
    float * V1 = (float *) &U1[trueP];
    float * U2 = (float *) &V1[trueP];
    float * V2 = (float *) &U2[trueP];
    float * W = (float *) &V2[trueP];
    float * X = (float *) &W[trueP];
    float * Y = (float *) &X[trueP];
    float * Z = (float *) &Y[trueP];

    // for i in range(trueP):
    //     for j in range(trueP):
    //         a[i][j] += u1[i] * v1[j] + u2[i] * v2[j]
    cblas_sger(CblasRowMajor, trueP, trueP, 1.0f, U1, 1, V1, 1, A, trueP);
    cblas_sger(CblasRowMajor, trueP, trueP, 1.0f, U2, 1, V2, 1, A, trueP);

    // for i in range(trueP):
    //     for j in range(trueP):
    //         x[i] = x[i] + -1.1 * a[j][i] * y[j]
    cblas_sgemv(CblasRowMajor, CblasTrans, trueP, trueP, beta, A, trueP, Y, 1, 1.0f, X, 1);

    // for i in range(trueP):
    //     x[i] += z[i]
    cblas_saxpy(trueP, 1.0f, Z, 1, X, 1);
    
    // for i in range(trueP):
    //     for j in range(trueP):
    //         w[i] = w[i] + 4.5 * a[i][j] * x[j]
    cblas_sgemv(CblasRowMajor, CblasNoTrans, trueP, trueP, alpha, A, trueP, X, 1, 1.0f, W, 1);
}
