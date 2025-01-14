from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class ComputeBaseGenerator(Generator):
    def __init__(self, experiment, kernel_name, testing=False):
        super().__init__(experiment, kernel_name)
        self.testing = testing

    def get_remainder(self, configuration, trueConfiguration):
        return ((np.random.randint(100, size=(self.get_size_to_allocate_i(configuration) - self.get_size_to_allocate_i(trueConfiguration))) / 10) - 5)

    def initialize_zero(self, cc, W, base_register, stride_unrolls, portion_unrolls):
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        
        n_regs = min(stride_unrolls * portion_unrolls, self.experiment.constants.n_simd_regs)
        for i in range(n_regs):
            cc.get_register(register_set="simd", variable=f"zero{i}")
        first = f"%{cc.get_variable('zero0')}"

        cc.add_statement("vxorps", first, first, first)
        for i in range(1, n_regs):
            src = f"%{cc.get_variable(f'zero{i-1}')}"
            dst = f"%{cc.get_variable(f'zero{i}')}"
            cc.add_statement("vmovaps", src, dst)

        with For(cc, f"${W // portion_unroll_values}"):
            zero_registers_pos = 0
            for j in range(portion_unrolls):
                for i in range(stride_unrolls):
                    reg = f"%{cc.get_variable(f'zero{zero_registers_pos}')}"
                    offset = self.aligned_z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                    cc.add_statement("vmovaps", reg, f"{offset}({base_register})")
                    zero_registers_pos = (zero_registers_pos + 1) % n_regs
            cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", base_register)
        
        for i in range(n_regs):
            cc.unset_variable(variable=f"zero{i}")
        
        cc.add_statement("subq", f"${self.aligned(W * self.experiment.constants.dtype_size_bytes)}", base_register)

    def writeback(self, cc, W, I, O, stride_unrolls, portion_unrolls):
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes

        with For(cc, f"${W // portion_unroll_values}"):
            for j in range(portion_unrolls):
                for i in range(stride_unrolls):
                    reg = cc.get_register(register_set='simd', variable='vec')
                    offset = self.aligned_z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                    cc.add_statement("vmovaps", f"{offset}({I})", f"%{reg}")
                    cc.add_statement("vmovaps", f"%{reg}", f"{offset}({O})")
                    cc.unset_variable('vec')

            cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", I)
            cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", O)

        cc.add_statement("subq", f"${self.aligned(W * self.experiment.constants.dtype_size_bytes)}", I)
        cc.add_statement("subq", f"${self.aligned(W * self.experiment.constants.dtype_size_bytes)}", O)

