from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import MxVBaseGenerator

class MxVGenerator(MxVBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "mxv")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P + P * P
    
    def get_size_to_allocate_i(self, configuration):
        return MxVGenerator.get_size_to_allocate(configuration)

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
            B = f"%{cc.get_register(variable='B')}"
            C = f"%{cc.get_register(variable='C')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, B)
            cc.add_statement("addq", offset, B)

            cc.add_statement("movq", D, C)
            cc.add_statement("addq", offset, C)
            cc.add_statement("addq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", C)

            cc.add_statement("movq", stack_ptr, frame_ptr)
            cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
            cc.add_statement("andq", f"$-{self.experiment.constants.simd_vec_bytes}", stack_ptr)      

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
                            offset_c = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                            offset_b = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            
                            ymm_c = f"%{cc.get_register(register_set='simd', variable='c')}"
                            ymm_b = f"%{cc.get_register(register_set='simd', variable='b')}"

                            cc.add_statement("vmovaps", f"{offset_c}({stack_ptr})", ymm_c)
                            cc.add_statement("vmovaps", f"{offset_b}({B})", ymm_b)
                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", ymm_b, ymm_c)
                            cc.add_statement("vmovaps", ymm_c, f"{offset_c}({stack_ptr})")

                            cc.unset_variable("c")
                            cc.unset_variable("b")
                    
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", B)
                
                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", B)
                cc.add_statement("addq", f"${self.aligned(stride_unrolls * self.experiment.constants.simd_vec_bytes)}", stack_ptr)

            cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
            cc.add_statement("subq", offset, D)

            if self.testing:
                ymm_alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"
                ymm_beta = f"%{cc.get_register(register_set='simd', variable='beta')}"
                xmm_beta = f"%{cc.get_variable('beta', size_column=2)}"

                cc.add_statement("movl", "$1083179008", f"-8({stack_ptr})")
                cc.add_statement("movl", "$3213675725", f"-16({stack_ptr})")

                cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_alpha)
                cc.add_statement("vbroadcastss", f"-16({stack_ptr})", ymm_beta)

                with For (cc, f"${trueP}"):
                    c_vec_y = f"%{cc.get_register(register_set='simd', variable='c')}"
                    c_vec_x = f"%{cc.get_variable('c', size_column=2)}"
                    ancilla = f"%{cc.get_register(register_set='simd', variable='ancilla_vec', size_column=2)}"
                    
                    cc.add_statement("vmovups", f"({stack_ptr})", c_vec_y)
                    cc.add_statement("vmulps", c_vec_y, ymm_alpha, c_vec_y)
                    cc.add_statement("vextractf128", "$1", c_vec_y, ancilla)
                    cc.add_statement("vaddps", c_vec_x, ancilla, c_vec_x)
                    cc.add_statement("vmovhlps", c_vec_x, c_vec_x, ancilla)
                    cc.add_statement("vaddps", c_vec_x, ancilla, c_vec_x)
                    cc.add_statement("vshufps", "$85", c_vec_x, c_vec_x, ancilla)
                    cc.add_statement("vaddss", ancilla, c_vec_x, c_vec_x)
                    cc.add_statement("vmulps", f"({C})", xmm_beta, ancilla)
                    cc.add_statement("vaddss", ancilla, c_vec_x, c_vec_x)
                    cc.add_statement("vmovss", c_vec_x, f"({C})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", stack_ptr)
                    cc.add_statement("addq", f"${self.experiment.constants.dtype_size_bytes}", C)
                    
                    cc.unset_variable("c")
                    cc.unset_variable("ancilla_vec")     

                cc.unset_variable("alpha")
                cc.unset_variable("beta")

            cc.add_statement("movq", frame_ptr, stack_ptr)

            cc.unset_variable('D')
            cc.unset_variable('frame_ptr')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('B')
            cc.unset_variable('C')