from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverMxVTBaseGenerator

class GemverMxVGenerator(GemverMxVTBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemvermxv")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P + P * P
    
    def get_size_to_allocate_i(self, configuration):
        return GemverMxVGenerator.get_size_to_allocate(configuration)

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

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            frame_ptr = f"%{cc.set_variable('rbp', variable='frame_ptr')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            A = f"%{cc.get_register(variable='A')}"
            W = f"%{cc.get_register(variable='W')}"
            X = f"%{cc.get_register(variable='X')}"
            ymm_alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, A)
            cc.add_statement("movq", D, W)
            cc.add_statement("addq", offset, W)

            cc.add_statement("leaq", f"{self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}({W})", X)

            cc.add_statement("movq", stack_ptr, frame_ptr)
            cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
            cc.add_statement("andq", f"$-{self.experiment.constants.simd_vec_bytes}", stack_ptr)

            cc.add_statement("movl", "$1083179008", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_alpha)

            if self.testing:
                zero_vec = f"%{cc.get_register(register_set='simd', variable='zero_vec')}"
                
                cc.add_statement("vxorps", zero_vec, zero_vec, zero_vec)
                with For (cc, f"${trueP}"):
                    cc.add_statement("vmovntdq", zero_vec, f"({stack_ptr})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", stack_ptr)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.simd_vec_bytes)}", stack_ptr)

                cc.unset_variable("zero_vec")
            
            with For(cc, f"${trueP // stride_unrolls}"):
                with For(cc, f"${trueP // portion_unroll_values}"):
                    for j in range(portion_unrolls):
                        for i in range(stride_unrolls):
                            offset_w = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                            offset_x = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            
                            ymm_w = f"%{cc.get_register(register_set='simd', variable='w')}"
                            ymm_x = f"%{cc.get_register(register_set='simd', variable='x')}"

                            cc.add_statement("vmovaps", f"{offset_w}({stack_ptr})", ymm_w)
                            cc.add_statement("vmulps", f"{offset_x}({X})", ymm_alpha, ymm_x)
                            cc.add_statement("vfmadd231ps", f"{offset_a}({A})", ymm_x, ymm_w)
                            cc.add_statement("vmovaps", ymm_w, f"{offset_w}({stack_ptr})")

                            cc.unset_variable("w")
                            cc.unset_variable("x")
                    
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", A)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", X)
                
                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", A)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", X)
                cc.add_statement("addq", f"${self.aligned(stride_unrolls * self.experiment.constants.simd_vec_bytes)}", stack_ptr)

            if self.testing:
                cc.add_statement("subq",f"${self.aligned(trueP * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
                cc.add_statement("subq", offset, A)

                with For (cc, f"${trueP}"):
                    w_vec_y = f"%{cc.get_register(register_set='simd', variable='w')}"
                    w_vec_x = f"%{cc.get_variable('w', size_column=2)}"
                    ancilla = f"%{cc.get_register(register_set='simd', variable='ancilla_vec', size_column=2)}"

                    cc.add_statement("vmovups", f"({stack_ptr})", w_vec_y)
                    cc.add_statement("vextractf128", "$1", w_vec_y, ancilla)
                    cc.add_statement("vaddps", w_vec_x, ancilla, w_vec_x)
                    cc.add_statement("vmovhlps", w_vec_x, w_vec_x, ancilla)
                    cc.add_statement("vaddps", w_vec_x, ancilla, w_vec_x)
                    cc.add_statement("vshufps", "$85", w_vec_x, w_vec_x, ancilla)
                    cc.add_statement("vaddss", ancilla, w_vec_x, w_vec_x)
                    cc.add_statement("vbroadcastss", f"({W})", ancilla)
                    cc.add_statement("vaddss", ancilla, w_vec_x, w_vec_x)
                    cc.add_statement("vmovss", w_vec_x, f"({W})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", stack_ptr)
                    cc.add_statement("addq", f"${self.experiment.constants.dtype_size_bytes}", W)
                    
                    cc.unset_variable("w")
                    cc.unset_variable("ancilla_vec")

            cc.add_statement("movq", frame_ptr, stack_ptr)

            cc.unset_variable('D')
            cc.unset_variable('frame_ptr')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('A')
            cc.unset_variable('W')
            cc.unset_variable('X')
            cc.unset_variable('alpha')