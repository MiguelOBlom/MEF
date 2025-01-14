#include "util.h"
#include "stdio.h"

const float alpha = 4.5f;
const float beta = -1.1f;

long unsigned int N = trueP + trueP + trueP * trueP;

void experiment (float * Data) {
    float (*A)[trueP] = (float (*)[trueP]) &Data[0];
    float (*B) = (float (*)) &A[trueP][0];
    float (*C) = (float (*)) &B[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        C[i] *= beta;

        for (unsigned int j = 0; j < trueP; j++) {
            C[i] += alpha * A[i][j] * B[j];
        }
    }
}
