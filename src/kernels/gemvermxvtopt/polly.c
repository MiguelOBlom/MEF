#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * trueP + trueP * trueP;

void experiment (float * Data) {
    float Beta = -1.1f;

    float (*A)[trueP] = (float (*)[trueP]) &Data[0];
    float (*X) = (float (*)) &A[trueP][0];
    float (*Y) = (float (*)) &X[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            X[i] = X[i] + Beta * A[j][i] * Y[j];
        }
    }
}
