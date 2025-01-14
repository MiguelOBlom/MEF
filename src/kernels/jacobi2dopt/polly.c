#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * trueX * trueX;
const float alpha = 0.2f;

void experiment (float * Data) {
    float (*A)[trueX] = (float (*)[trueX]) &Data[0];
    float (*B)[trueX] = (float (*)[trueX]) &A[trueX][0];

    for (unsigned int s = 0; s < S; s++){
        for (unsigned int i = 1; i < trueX - 1; i++) {
            for (unsigned int j = 1; j < trueX - 1; j++) {
                B[i][j] = alpha * (A[i - 1][j] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j]);
            }
        }

        for (unsigned int i = 1; i < trueX - 1; i++) {
            for (unsigned int j = 1; j < trueX - 1; j++) {
                A[i][j] = B[i][j];
            }
        }
    }
}
