#include "util.h"
#include "stdio.h"

long unsigned int N = 4 * trueP + trueP * trueP;

void experiment (float * Data) {
    float (*A)[trueP] = (float (*)[trueP]) &Data[0];
    float (*U1) = (float (*)) &A[trueP][0];
    float (*V1) = (float (*)) &U1[trueP];
    float (*U2) = (float (*)) &V1[trueP];
    float (*V2) = (float (*)) &U2[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        for (unsigned int j = 0; j < trueP; j++) {
            A[i][j] = A[i][j] + U1[i] * V1[j] + U2[i] * V2[j];
        }
    }
}
