from .base import BicGBaseGenerator
from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class BicGOptGenerator(BicGBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "bicgopt", testing=testing)

    @staticmethod
    def get_size_to_allocate(configuration):
        X = configuration["X"]
        return X * X + 4 * X

    def get_size_to_allocate_i(self, configuration):
        return BicGOptGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        X = configuration["X"]
        N = self.get_size_to_allocate_i(configuration)

        stride_unrolls_init = configuration["stride_unrolls_init"]
        portion_unrolls_init = configuration["portion_unrolls_init"]
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]

        # reserve unroll_i registers for q
        # reserve unroll_i registers for r
        # reserve at least one register for s
        # reserve at least one register for p
        # reserve at least one register for a <- can be used in writeback
        if 2 * stride_unrolls + 3 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        portion_unroll_init_values = portion_unrolls_init * self.experiment.constants.simd_vec_values
        portion_unroll_init_bytes = portion_unrolls_init * self.experiment.constants.simd_vec_bytes
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X, [stride_unrolls_init, stride_unrolls], [portion_unrolls_init, portion_unrolls], unalignment_factor=unalignment_factor)

        if trueX is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = self.get_size_to_allocate_i({"X": trueX})

        W_init = trueX // stride_unrolls_init

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"], test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            S = f"%{cc.get_register(variable='S')}"
            Q = f"%{cc.get_register(variable='Q')}"
            R = f"%{cc.get_register(variable='R')}"

            cc.add_statement("movq", f"${trueX}", offset)
            cc.add_statement("imul", f"${trueX * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueX * trueX * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, S)
            cc.add_statement("addq", offset, S)

            cc.add_statement("leaq", f"{self.aligned(trueX * self.experiment.constants.dtype_size_bytes)}({D})", Q)
            cc.add_statement("addq", offset, Q)

            cc.add_statement("leaq", f"{self.aligned(3 * trueX * self.experiment.constants.dtype_size_bytes)}({D})", R)
            cc.add_statement("addq", offset, R)

            self.initialize_zero(cc, W_init, S, stride_unrolls_init, portion_unrolls_init)

            # Compute BicG
            with For(cc, f"${trueX // stride_unrolls}"):
                # Initialize Q
                for i in range(stride_unrolls):
                    cc.get_register(register_set="simd", variable=f"q{i}")
                first = f"%{cc.get_variable('q0')}"

                cc.add_statement("vxorps", first, first, first)
                for i in range(1, stride_unrolls):
                    src = f"%{cc.get_variable(f'q{i-1}')}"
                    dst = f"%{cc.get_variable(f'q{i}')}"
                    cc.add_statement("vmovaps", src, dst)

                # Initialize R
                for i in range(stride_unrolls):
                    r_register = cc.get_register(register_set="simd", variable=f"r{i}")
                    offset_broadcast = self.z(i * self.experiment.constants.dtype_size_bytes) 
                    cc.add_statement("vbroadcastss", f"{offset_broadcast}({R})", f"%{r_register}")
                
                # Compute loop
                with For(cc, f"${trueX // portion_unroll_values}"):
                    for j in range(portion_unrolls):
                        offset_s = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                        offset_p = self.aligned_z(2 * trueX * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)

                        ymm_s = f"%{cc.get_register(register_set='simd', variable='s')}"
                        ymm_p = f"%{cc.get_register(register_set='simd', variable='p')}"

                        cc.add_statement("vmovaps", f"{offset_s}({S})", ymm_s)
                        cc.add_statement("vmovaps", f"{offset_p}({S})", ymm_p)

                        for i in range(stride_unrolls):
                            ymm_q = f"%{cc.get_variable(variable=f'q{i}')}"
                            ymm_r = f"%{cc.get_variable(variable=f'r{i}')}"
                            ymm_a = f"%{cc.get_register(register_set='simd', variable='a')}"

                            offset_a = self.aligned_z(i * trueX * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            
                            cc.add_statement("vmovaps", f"{offset_a}({D})", ymm_a)
                            cc.add_statement("vfmadd231ps", ymm_r, ymm_a, ymm_s)
                            cc.add_statement("vfmadd231ps", ymm_p, ymm_a, ymm_q)

                            cc.unset_variable("a")

                        cc.add_statement("vmovaps", ymm_s, f"{offset_s}({S})")

                        cc.unset_variable("s")
                        cc.unset_variable("p")

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", S)

                for i in range(stride_unrolls):
                    cc.unset_variable(f"r{i}")

                # Write back Q
                for i in range(stride_unrolls):
                    q_register = f"%{cc.get_variable(variable=f'q{i}')}"
                    q_register_x = f"%{cc.get_variable(variable=f'q{i}', size_column=2)}"
                    
                    ancilla = f"%{cc.get_register(register_set='simd', variable='ancilla', size_column=2)}"
                    
                    offset_q = self.z(i * self.experiment.constants.dtype_size_bytes)
                    cc.add_statement("vextractf128", "$0x1", q_register, ancilla)
                    cc.add_statement("vaddps", q_register_x, ancilla, ancilla)
                    cc.add_statement("vhaddps", ancilla, ancilla, ancilla)
                    cc.add_statement("vhaddps", ancilla, ancilla, ancilla)
                    cc.add_statement("vmovss", ancilla, f"{offset_q}({Q})")
                    
                    cc.unset_variable(f'q{i}')
                    cc.unset_variable('ancilla')
                
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", Q)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", R)
                cc.add_statement("addq", f"${(trueX * (stride_unrolls - 1)) * self.experiment.constants.dtype_size_bytes}", D)
                cc.add_statement("subq", f"${trueX * self.experiment.constants.dtype_size_bytes}", S)

            cc.add_statement("subq", offset, D)

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('S')
            cc.unset_variable('Q')
            cc.unset_variable('R')
        return (trueN, N, {"trueX": trueX})