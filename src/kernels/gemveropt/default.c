#include "util.h"
#include "stdio.h"

long unsigned int N = 8 * trueP + trueP * trueP;

const float alpha = 4.5f;
const float beta = -1.1f;

void experiment (float * Data) {
    float (*A)[trueP] = (float (*)[trueP]) &Data[0];
    float (*U1) = (float (*)) &A[trueP][0];
    float (*V1) = (float (*)) &U1[trueP];
    float (*U2) = (float (*)) &V1[trueP];
    float (*V2) = (float (*)) &U2[trueP];
    float (*W) = (float (*)) &V2[trueP];
    float (*X) = (float (*)) &W[trueP];
    float (*Y) = (float (*)) &X[trueP];
    float (*Z) = (float (*)) &Y[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            A[i][j] = A[i][j] + U1[i] * V1[j] + U2[i] * V2[j];
        }
    }

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            X[i] = X[i] + beta * A[j][i] * Y[j];
        }
    }

    for (unsigned int i = 0; i < trueP; i++) {
        X[i] = X[i] + Z[i];
    }

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            W[i] = W[i] + alpha * A[i][j] * X[j];
        }
    }
}
