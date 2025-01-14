from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverMxVTBaseGenerator

class GemverMxVOptGenerator(GemverMxVTBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemvermxvopt")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return GemverMxVOptGenerator.get_size_to_allocate(configuration)

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
            W = f"%{cc.get_register(variable='W')}"
            X = f"%{cc.get_register(variable='X')}"
            ymm_alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, W)
            cc.add_statement("addq", offset, W)

            cc.add_statement("leaq", f"{self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}({W})", X)

            cc.add_statement("movl", "$1083179008", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_alpha)
           
            with For(cc, f"${trueP // stride_unrolls}"):
                for i in range(stride_unrolls):
                    cc.get_register(register_set="simd", variable=f"w{i}")
                first = f"%{cc.get_variable('w0')}"

                cc.add_statement("vxorps", first, first, first)
                for i in range(1, stride_unrolls):
                    src = f"%{cc.get_variable(f'w{i-1}')}"
                    dst = f"%{cc.get_variable(f'w{i}')}"
                    cc.add_statement("vmovaps", src, dst)

                with For(cc, f"${trueP // portion_unroll_values}"):
                    for j in range(portion_unrolls):
                        offset_x = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                        ymm_x = cc.get_register(register_set='simd', variable=f'x{j}')

                        cc.add_statement("vmulps", f"{offset_x}({X})", ymm_alpha, f"%{ymm_x}")

                        for i in range(stride_unrolls):
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            ymm_w = f"%{cc.get_variable(f'w{i}')}"

                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", f"%{ymm_x}", ymm_w)

                        cc.unset_variable(f'x{j}')

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", X)

                for i in range(stride_unrolls):
                    offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                    ymm_w = cc.get_variable(f'w{i}')
                    xmm_w = cc.get_variable(f'w{i}', size_column=2)
                    ancilla = cc.get_register(register_set='simd', variable='ancilla', size_column=2)

                    offset_w = self.z(i * self.experiment.constants.dtype_size_bytes)

                    cc.add_statement("vextractf128", "$0x1", f"%{ymm_w}", f"%{ancilla}")
                    cc.add_statement("vaddps", f"%{xmm_w}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vhaddps", f"%{ancilla}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vhaddps", f"%{ancilla}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vbroadcastss", f"{offset_w}({W})", f"%{xmm_w}")
                    cc.add_statement("vaddss", f"%{xmm_w}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vmovss", f"%{ancilla}", f"{offset_w}({W})")

                    cc.unset_variable(f'w{i}')
                    cc.unset_variable('ancilla')


                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", X)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", W)
                
            cc.add_statement("subq", offset, D)

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('W')
            cc.unset_variable('X')
            cc.unset_variable('alpha')
        return (trueN, N, {"trueP": trueP})