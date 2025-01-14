#include "util.h"
#include "jacobi2dopthalide.h"
#include "writebackhalide.h"

#include "HalideBuffer.h"

long unsigned int N = 2 * trueX * trueX;

// clang -c -std=c++17 -O3 -Wall -Wextra -I../Halide/include -I../Halide/tools -I/usr/include/libpng16 -Isrc/utils -Isrc/utils/perf_utils/include -DtrueX=16384 -DPAGE_SIZE=4096 -DREPETITIONS=5 -fno-inline -o kernels/jacobi2dopt/bin_ctrl/halide_2117442904_4096.o src/kernels/jacobi2dopt/halide.cpp; clang -O3 -Wall -Wextra -I../Halide/include -I../Halide/tools -I/usr/include/libpng16 -Isrc/utils -Isrc/utils/perf_utils/include -L../Halide/bin -DtrueX=16384 -DPAGE_SIZE=4096 -DREPETITIONS=5 -fno-inline -o kernels/jacobi2dopt/bin_ctrl/halide_2117442904_4096 kernels/jacobi2dopt/bin_ctrl/halide_2117442904_4096.o src/main.c src/utils/perf_utils/obj/utils.o -lHalide -lpng16 -ljpeg -lpthread -ldl -lstdc++
// LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../Halide/bin kernels/jacobi2dopt/bin_ctrl/halide_2117442904_4096

void experiment (float * Data) {
    float * A = &Data[0];
    float * B = &A[trueX * trueX];

    Halide::Runtime::Buffer<float> input1{&A[0], trueX, trueX};
    Halide::Runtime::Buffer<float> output1(&B[0], trueX - 2, trueX - 2);
    
    Halide::Runtime::Buffer<float> output2(&B[0], trueX, trueX);
    Halide::Runtime::Buffer<float> input2{&A[0], trueX - 2, trueX - 2};

    for (unsigned int s = 0; s < S; s++) {
        jacobi2dopthalide(input1, output1);
        writebackhalide(output2, input2);
    }

}