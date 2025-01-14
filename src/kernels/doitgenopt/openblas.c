#include "util.h"
#include "stdio.h"
#include <cblas.h>

long unsigned int N = 2 * (trueR * trueQ * trueP) + (trueP * trueP);

void experiment (float * Data) {
    float (*A)[trueQ][trueP] = (float (*)[trueQ][trueP]) &Data[0];
    float (*C4)[trueP] = (float (*)[trueP]) &A[trueR][0][0];
    float (*Sum)[trueQ][trueP] = (float (*)[trueQ][trueP]) &C4[trueP][0];
    
    for (int r = 0; r < trueR; r++) {
        for (int q = 0; q < trueQ; q++) {
            cblas_sgemv(CblasRowMajor, CblasTrans, trueP, trueP, 1.0, &C4[0][0], trueP, A[r][q], 1, 0.0, Sum[r][q], 1);
            cblas_scopy(trueR, Sum[r][q], 1, A[r][q], 1);
        }
    }
}