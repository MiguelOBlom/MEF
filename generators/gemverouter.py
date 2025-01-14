from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverOuterBaseGenerator

class GemverOuterGenerator(GemverOuterBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemverouter")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 4 * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return GemverOuterGenerator.get_size_to_allocate(configuration)

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
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            U1 = f"%{cc.get_register(variable='U1')}"
            V1 = f"%{cc.get_register(variable='V1')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, U1)
            cc.add_statement("addq", offset, U1)
            
            cc.add_statement("leaq", f"{self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}({U1})", V1)

            with For(cc, f"${trueP // stride_unrolls}"):
                with For(cc, f"${trueP // portion_unroll_values}"):
                    for j in range(portion_unrolls):
                        for i in range(stride_unrolls):
                            offset_u1 = self.z(i * self.experiment.constants.dtype_size_bytes)
                            offset_u2 = self.z(i * self.experiment.constants.dtype_size_bytes + 2 * trueP * self.experiment.constants.dtype_size_bytes)
                            offset_v1 = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                            offset_v2 = self.aligned_z(j * self.experiment.constants.simd_vec_bytes + 2 * trueP * self.experiment.constants.dtype_size_bytes)
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)

                            ymm_u1 = f"%{cc.get_register(register_set='simd', variable='u1')}"
                            ymm_u2 = f"%{cc.get_register(register_set='simd', variable='u2')}"
                            ymm_a = f"%{cc.get_register(register_set='simd', variable='a')}"

                            cc.add_statement("vbroadcastss", f"{offset_u1}({U1})", ymm_u1)
                            cc.add_statement("vbroadcastss", f"{offset_u2}({U1})", ymm_u2)
                            cc.add_statement("vmovaps", f"{offset_a}({D})", ymm_a)
                            cc.add_statement("vfmadd231ps", f"{offset_v1}({V1})", ymm_u1, ymm_a)
                            cc.add_statement("vfmadd231ps", f"{offset_v2}({V1})", ymm_u2, ymm_a)
                            cc.add_statement("vmovaps", ymm_a, f"{offset_a}({D})")

                            cc.unset_variable("u1")
                            cc.unset_variable("u2")
                            cc.unset_variable("a")
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D);
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", V1);
                cc.add_statement("addq", f"${self.aligned((stride_unrolls - 1) * trueP * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", U1)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", V1)
            
            cc.add_statement("subq", offset, D)
            
            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('U1')
            cc.unset_variable('V1')