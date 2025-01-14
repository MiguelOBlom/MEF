from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import Jacobi2DBaseGenerator

class Jacobi2DGenerator(Jacobi2DBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "jacobi2d")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        X = configuration["X"]
        return 2 * X * X
    
    def get_size_to_allocate_i(self, configuration):
        return Jacobi2DGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        X = configuration["X"]
        N = self.get_size_to_allocate_i(configuration)
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X - 2, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)

        if trueX is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueX += 2


        trueN = self.get_size_to_allocate_i({"X": trueX})


        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            
            O = f"%{cc.get_register(variable='output')}"
            I = f"%{cc.get_register(variable='input')}"
            
            alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"

            cc.add_statement("movq", f"${trueX}", offset)
            cc.add_statement("imul", f"${trueX * self.experiment.constants.dtype_size_bytes}", offset)

            cc.add_statement("movl", "$1045220557", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", alpha)

            cc.add_statement("movq", D, I)
            cc.add_statement("movq", D, O)
            cc.add_statement("addq", offset, O)

            with For(cc, f"${(trueX - 2) // stride_unrolls}"):
                with For(cc, f"${(trueX - 2) // portion_unroll_values}"):
                    for i in range(portion_unrolls):
                        for j in range(stride_unrolls):
                            ymm = f"%{cc.get_register(register_set='simd', variable='vec')}"

                            offset_u = self.z(j * trueX * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes + self.experiment.constants.dtype_size_bytes)
                            offset_c = self.z(j * trueX * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes + self.experiment.constants.dtype_size_bytes + trueX * self.experiment.constants.dtype_size_bytes)
                            offset_d = self.z(j * trueX * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes + self.experiment.constants.dtype_size_bytes + 2 * trueX * self.experiment.constants.dtype_size_bytes)
                            offset_l = self.z(j * trueX * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes + trueX * self.experiment.constants.dtype_size_bytes)
                            offset_r = self.z(j * trueX * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes + 2 * self.experiment.constants.dtype_size_bytes + trueX * self.experiment.constants.dtype_size_bytes)

                            cc.add_statement("vmovups", f"{offset_u}({I})", ymm)
                            cc.add_statement("vaddps", f"{offset_l}({I})", ymm, ymm)
                            cc.add_statement("vaddps", f"{offset_c}({I})", ymm, ymm)
                            cc.add_statement("vaddps", f"{offset_r}({I})", ymm, ymm)
                            cc.add_statement("vaddps", f"{offset_d}({I})", ymm, ymm)
                            cc.add_statement("vmulps", ymm, alpha, ymm)
                            cc.add_statement("vmovups", ymm, f"{offset_c}({O})")

                            cc.unset_variable("vec")

                    cc.add_statement("addq",f"${portion_unroll_bytes}", I)
                    cc.add_statement("addq",f"${portion_unroll_bytes}", O)

                cc.add_statement("addq", f"${((stride_unrolls - 1) * trueX + 2) * self.experiment.constants.dtype_size_bytes}", I)
                cc.add_statement("addq", f"${((stride_unrolls - 1) * trueX + 2) * self.experiment.constants.dtype_size_bytes}", O)

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('output')
            cc.unset_variable('input')
            cc.unset_variable('alpha')


    def test(self, configuration, test_data_dir):
        X = configuration["X"]
        S = configuration["S"]
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X - 2, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor) + 2
        
        a = ((np.random.randint(100, size=(trueX, trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        b = ((np.random.randint(100, size=(trueX, trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = (self.get_remainder(configuration, {"X": trueX})).astype(self.experiment.constants.nd_type)
        
        self.write_test_input(test_data_dir, a, b, remainder)

        alpha = 0.2
        for i in range(1, trueX - 1):
            for j in range(1, trueX - 1):
                b[i][j] = alpha * (a[i][j] + a[i + 1][j] + a[i - 1][j] + a[i][j + 1] + a[i][j - 1])

        self.write_test_output(test_data_dir, a, b, remainder)

        return test_data_dir