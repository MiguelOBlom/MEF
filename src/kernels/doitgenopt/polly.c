#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * (trueR * trueQ * trueP) + (trueP * trueP);

void experiment (float * Data) {
    float (*A)[trueQ][trueP] = (float (*)[trueQ][trueP]) &Data[0];
    float (*C4)[trueP] = (float (*)[trueP]) &A[trueR][0][0];
    float (*Sum)[trueQ][trueP] = (float (*)[trueQ][trueP]) &C4[trueP][0];

    for (unsigned int r = 0; r < trueR; r++) {
        for (unsigned int q = 0; q < trueQ; q++) {
            for (unsigned int p = 0; p < trueP; p++) {
                Sum[r][q][p] = 0;
                for (unsigned int s = 0; s < trueP; s++) {
                    Sum[r][q][p] = Sum[r][q][p] + A[r][q][s] * C4[s][p];
                }
            }
            for (unsigned int p = 0; p < trueR; p++) {
                A[r][q][p] = Sum[r][q][p];
            }
        }
    }
}

