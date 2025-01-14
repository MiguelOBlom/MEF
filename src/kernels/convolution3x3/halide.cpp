#include "util.h"
#include "convolution3x3halide.h"

#include "HalideBuffer.h"

long unsigned int N = 2 * trueX * trueX + 16;

// clang -c -std=c++17 -O3 -Wall -Wextra -I../Halide/include -I../Halide/tools -I/usr/include/libpng16 -Isrc/utils -Isrc/utils/perf_utils/include -DtrueX=16384 -DPAGE_SIZE=4096 -DREPETITIONS=5 -fno-inline -o kernels/conv/bin_ctrl/halide_2117442904_4096.o src/kernels/conv/halide.cpp; clang -O3 -Wall -Wextra -I../Halide/include -I../Halide/tools -I/usr/include/libpng16 -Isrc/utils -Isrc/utils/perf_utils/include -L../Halide/bin -DtrueX=16384 -DPAGE_SIZE=4096 -DREPETITIONS=5 -fno-inline -o kernels/conv/bin_ctrl/halide_2117442904_4096 kernels/conv/bin_ctrl/halide_2117442904_4096.o src/main.c src/utils/perf_utils/obj/utils.o -lHalide -lpng16 -ljpeg -lpthread -ldl -lstdc++
// LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../Halide/bin kernels/conv/bin_ctrl/halide_2117442904_4096

void experiment (float * Data) {
    float *C = &Data[0];
    float *A = &C[16];
    float *B = &A[trueX * trueX];

    Halide::Runtime::Buffer<float> input{&A[0], trueX, trueX};
    Halide::Runtime::Buffer<float> kernel{&C[0], 3, 3};
    Halide::Runtime::Buffer<float> output(&B[0], trueX - 2, trueX - 2);
    convolution3x3halide(input, kernel, output);

}