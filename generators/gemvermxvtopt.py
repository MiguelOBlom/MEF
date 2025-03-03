from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverMxVBaseGenerator

class GemverMxVTOptGenerator(GemverMxVBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemvermxvtopt")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return GemverMxVTOptGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        P = configuration["P"]

        N = self.get_size_to_allocate_i(configuration)

        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        
        if trueP is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = self.get_size_to_allocate_i({"P": trueP})

        if stride_unrolls + 2 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"], test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            X = f"%{cc.get_register(variable='X')}"
            Y = f"%{cc.get_register(variable='Y')}"
            ymm_beta = f"%{cc.get_register(register_set='simd', variable='beta')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, X)
            cc.add_statement("addq", offset, X)
            cc.add_statement("leaq", f"{self.aligned_z(trueP * self.experiment.constants.dtype_size_bytes)}({X})", Y)

            cc.add_statement("movl", "$3213675725", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_beta)

            with For(cc, f"${trueP // stride_unrolls}"):
                for j in range(stride_unrolls):
                    ymm_y = f"%{cc.get_register(register_set='simd', variable=f'y{j}')}"
                    offset_y = self.z(j * self.experiment.constants.dtype_size_bytes)
                    cc.add_statement("vbroadcastss", f"{offset_y}({Y})", ymm_y)
                    cc.add_statement("vmulps", ymm_y, ymm_beta, ymm_y)

                with For(cc, f"${trueP // portion_unroll_values}"):
                    for i in range(portion_unrolls):
                        offset_x = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                        ymm_x = cc.get_register(register_set='simd', variable=f'x{j}')

                        cc.add_statement("vmovaps", f"{offset_x}({X})", f"%{ymm_x}")

                        for j in range(stride_unrolls):
                            ymm_y = f"%{cc.get_variable(f'y{j}')}"
                            offset_a = self.aligned_z(j * trueP * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes)
                          
                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", ymm_y, f"%{ymm_x}")

                        cc.add_statement("vmovaps", f"%{ymm_x}", f"{offset_x}({X})")

                        cc.unset_variable(f"x{j}")

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", X)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)

                for j in range(stride_unrolls):
                    cc.unset_variable(f'y{j}')

                cc.add_statement("addq", f"${self.aligned((stride_unrolls - 1) * trueP * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", X)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", Y)       

            cc.add_statement("subq", offset, D)

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('X')
            cc.unset_variable('Y')
            cc.unset_variable('beta')
        return (trueN, N, {"trueP": trueP})