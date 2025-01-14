#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// clang jacobi2dopt.cpp ../../Halide/tools/GenGen.cpp -std=c++17 -fno-rtti -DSIDE=16384 -I ../../Halide/include/ -L ../../Halide/bin/ -lHalide -lpthread -ldl -o jacobi2dopt -lstdc++
// ./jacobi2dopt -o . -g jacobi2dopt_auto_schedule_gen -f jacobi2dopthalide -e static_library,h,schedule -p ../../Halide/bin/libautoschedule_mullapudi2016.so target=host autoscheduler=Mullapudi2016 autoscheduler.parallelism=1 autoscheduler.last_level_cache_size=12582912


const float alpha = 0.2f;

class Jacobi2DAutoScheduled : public Halide::Generator<Jacobi2DAutoScheduled> {
public:
    Input<Buffer<float, 2> > input{"input"};
    Output<Buffer<float, 2> > output{"output"};
    Var x{"x"}, y{"y"};

    void generate() {
        output(x, y) = alpha * (input(x + 1, y + 1) + input(x, y + 1) + input(x + 2, y + 1) + input(x + 1, y) + input(x + 1, y + 2));
    }

    void schedule() {
        input.set_estimates({{0, SIDE}, {0, SIDE}});
        output.set_estimates({{1, SIDE - 1}, {1, SIDE - 1}});
    }

private:
};


HALIDE_REGISTER_GENERATOR(Jacobi2DAutoScheduled, jacobi2dopt_auto_schedule_gen)


class WriteBackAutoScheduled : public Halide::Generator<WriteBackAutoScheduled> {
public:
    Input<Buffer<float, 2> > output{"output"};
    Output<Buffer<float, 2> > input{"input"};
    Var x{"x"}, y{"y"};

    void generate() {
        input(x, y) = output(x, y);
    }

    void schedule() {
        output.set_estimates({{0, SIDE}, {0, SIDE}});
        input.set_estimates({{1, SIDE - 1}, {1, SIDE - 1}});
    }

private:
};


HALIDE_REGISTER_GENERATOR(WriteBackAutoScheduled, writeback_auto_schedule_gen)