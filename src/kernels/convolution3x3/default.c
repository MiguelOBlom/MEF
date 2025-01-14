#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * trueX * trueX + 16;

void experiment (float * Data) {
    float (*C) = (float (*)) &Data[0];
    float (*A)[trueX] = (float (*)[trueX]) &C[16];
    float (*B)[trueX] = (float (*)[trueX]) &A[trueX][0];

    for (unsigned int i = 1; i < trueX - 1; i++) {
        for (unsigned int j = 1; j < trueX - 1; j++) {
            B[i][j] = C[0] * A[i - 1][j - 1];
            B[i][j] += C[1] * A[i - 1][j];
            B[i][j] += C[2] * A[i - 1][j + 1];
            B[i][j] += C[3] * A[i][j - 1];
            B[i][j] += C[4] * A[i][j];
            B[i][j] += C[5] * A[i][j + 1];
            B[i][j] += C[6] * A[i + 1][j - 1];
            B[i][j] += C[7] * A[i + 1][j];
            B[i][j] += C[8] * A[i + 1][j + 1];
        }
    }
}
