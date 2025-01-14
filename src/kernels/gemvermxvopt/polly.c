#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * trueP + trueP * trueP;

void experiment (float * Data) {
    float Alpha = 4.5f;

    float (*A)[trueP] = (float (*)[trueP]) &Data[0];
    float (*W) = (float (*)) &A[trueP][0];
    float (*X) = (float (*)) &W[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            W[i] = W[i] + Alpha * A[i][j] * X[j];
        }
    }
}
