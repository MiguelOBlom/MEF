#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// clang conv.cpp ../../Halide/tools/GenGen.cpp -std=c++17 -fno-rtti -DN_=16384 -I ../../Halide/include/ -L ../../Halide/bin/ -lHalide -lpthread -ldl -o conv -lstdc++
// ./conv -o . -g conv_auto_schedule_gen -f convhalide -e static_library,h,schedule -p ../../Halide/bin/libautoschedule_mullapudi2016.so target=host autoscheduler=Mullapudi2016 autoscheduler.parallelism=1 autoscheduler.last_level_cache_size=12582912




class Convolution3x3AutoScheduled : public Halide::Generator<Convolution3x3AutoScheduled> {
public:
    Input<Buffer<float, 2> > input{"input"};
    Input<Buffer<float, 2> > kernel{"kernel"};
    RDom kernel_rdom{0, 3, 0, 3};
    Output<Buffer<float, 2> > output{"output"};
    Var x{"x"}, y{"y"};

    void generate() {
        output(x, y) = sum(kernel(kernel_rdom.x, kernel_rdom.y) * input(x + kernel_rdom.x, y + kernel_rdom.y));
    }

    void schedule() {
        input.set_estimates({{0, SIDE}, {0, SIDE}});
        kernel.set_estimates({{0, 3}, {0, 3}});
        output.set_estimates({{0, SIDE - 2}, {0, SIDE - 2}});
    }

private:
};


HALIDE_REGISTER_GENERATOR(Convolution3x3AutoScheduled, convolution3x3_auto_schedule_gen)
