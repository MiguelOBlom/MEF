from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import MxVBaseGenerator

class MxVOptGenerator(MxVBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "mxvopt")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return  2 * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return MxVOptGenerator.get_size_to_allocate(configuration)

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

        if stride_unrolls + 2 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        trueN = self.get_size_to_allocate_i({"P": trueP})

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"], test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
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
            
            cc.add_statement("movl", "$1083179008", f"-8({stack_ptr})")
            cc.add_statement("movl", "$3213675725", f"-16({stack_ptr})")

            with For(cc, f"${trueP // stride_unrolls}"):
                for i in range(stride_unrolls):
                    cc.get_register(register_set="simd", variable=f"c{i}")
                first = f"%{cc.get_variable('c0')}"

                cc.add_statement("vxorps", first, first, first)
                for i in range(1, stride_unrolls):
                    src = f"%{cc.get_variable(f'c{i-1}')}"
                    dst = f"%{cc.get_variable(f'c{i}')}"
                    cc.add_statement("vmovaps", src, dst)

                with For(cc, f"${trueP // portion_unroll_values}"):
                    for j in range(portion_unrolls):

                        offset_b = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                        ymm_b = f"%{cc.get_register(register_set='simd', variable='b')}"
                        cc.add_statement("vmovaps", f"{offset_b}({B})", ymm_b)
                        
                        for i in range(stride_unrolls):
                            ymm_c = f"%{cc.get_variable(f'c{i}')}"
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", ymm_b, ymm_c)
                        
                        cc.unset_variable("b")

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", B)

                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", B)

                for i in range(stride_unrolls):
                    if i < 2:
                        ymm_alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"
                        cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_alpha)
                    
                    if i == 0:
                        ancilla = f"%{cc.get_variable(f'alpha', size_column=2)}"
                    else:
                        ancilla = f"%{cc.get_register(register_set='simd', variable='ancilla', size_column=2)}"

                    ymm_c = f"%{cc.get_variable(f'c{i}')}"
                    xmm_c = f"%{cc.get_variable(f'c{i}', size_column=2)}"
                    
                    cc.add_statement("vmulps", ymm_c, ymm_alpha, ymm_c)
                    cc.add_statement("vextractf128", "$0x1", ymm_c, ancilla)
                    cc.add_statement("vaddps", xmm_c, ancilla, ancilla)
                    cc.add_statement("vhaddps", ancilla, ancilla, ancilla)
                    cc.add_statement("vhaddps", ancilla, ancilla, ancilla)

                    if i < 3:
                        if i == 0:
                            cc.unset_variable('alpha')
                        ymm_beta = f"%{cc.get_register(register_set='simd', variable='beta')}"
                        xmm_beta = f"%{cc.get_variable('beta', size_column=2)}"
                        cc.add_statement("vbroadcastss", f"-16({stack_ptr})", ymm_beta)

                    cc.add_statement("vmulps", f"{self.z(i * self.experiment.constants.dtype_size_bytes)}({C})", xmm_beta, xmm_c)
                    cc.add_statement("vaddss", ancilla, xmm_c, xmm_c)
                    cc.add_statement("vmovss", xmm_c, f"{self.z(i * self.experiment.constants.dtype_size_bytes)}({C})")

                    if i > 0:
                        cc.unset_variable('ancilla')
                    if i < 2:
                        cc.unset_variable('beta')
                    
                    cc.unset_variable(f'c{i}')

                if stride_unrolls > 1:
                    cc.unset_variable('alpha')
                if stride_unrolls > 2:
                    cc.unset_variable('beta')

                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", C)

            cc.add_statement("subq", offset, D)
            
            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('B')
            cc.unset_variable('C')
        return (trueN, N, {"trueP": trueP})