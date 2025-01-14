#include "util.h"
#include "stdio.h"

long unsigned int N = trueX * trueX + 2 * trueX + 2 * trueX;

void experiment (float * Data) {
    float (*A)[trueX] = (float (*)[trueX]) &Data[0];
    float (*S) = (float (*)) &A[trueX][0];
    float (*Q) = (float (*)) &S[trueX];
    float (*P) = (float (*)) &Q[trueX];
    float (*R) = (float (*)) &P[trueX];

    for (unsigned int i = 0; i < trueX; i++) {
        S[i] = 0;
    }

    for (unsigned int i = 0; i < trueX; i++) {
        Q[i] = 0;
        for(unsigned int j = 0; j < trueX; j++) {
            S[j] = S[j] + R[i] * A[i][j];
            Q[i] = Q[i] + A[i][j] * P[j];
        }
    }
}
