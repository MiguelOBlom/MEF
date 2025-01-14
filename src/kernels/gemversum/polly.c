#include "util.h"
#include "stdio.h"

long unsigned int N = 2 * trueP;

void experiment (float * Data) {
    float (*X) = (float (*)) &Data[0];
    float (*Z) = (float (*)) &X[trueP];

    for (unsigned int i = 0; i < trueP; i++) {
        X[i] = X[i] + Z[i];
    }
}
