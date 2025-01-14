from .base import BicGBaseGenerator
from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class BicGGenerator(BicGBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "bicg", testing=testing)

    @staticmethod
    def get_size_to_allocate(configuration):
        X = configuration["X"]
        return X * X + 4 * X

    def get_size_to_allocate_i(self, configuration):
        return BicGGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        X = configuration["X"]
        N = self.get_size_to_allocate_i(configuration)
        stride_unrolls = configuration["stride_unrolls"]
        stride_unroll_bytes = stride_unrolls * self.experiment.constants.simd_vec_bytes
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)

        if trueX is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = self.get_size_to_allocate_i({"X": trueX})

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            frame_ptr = f"%{cc.set_variable('rbp', variable='frame_ptr')}"
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
            cc.add_statement("movq", stack_ptr, frame_ptr)
            cc.add_statement("subq", f"${self.aligned(trueX * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
            cc.add_statement("andq", "$-32", stack_ptr)

            if self.testing:
                zero_vec = f"%{cc.get_register(register_set='simd', variable='zero_vec')}"
                
                cc.add_statement("vxorps", zero_vec, zero_vec, zero_vec)
                with For (cc, f"${trueX // self.experiment.constants.simd_vec_values}"):
                    cc.add_statement("vmovntdq", zero_vec, f"({S})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", S)
                cc.add_statement("subq", f"${self.aligned(trueX * self.experiment.constants.dtype_size_bytes)}", S)
                
                with For (cc, f"${trueX}"):
                    cc.add_statement("vmovntdq", zero_vec, f"({stack_ptr})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", stack_ptr)
                cc.add_statement("subq", f"${self.aligned(trueX * self.experiment.constants.simd_vec_bytes)}", stack_ptr)

                cc.unset_variable("zero_vec")

            with For(cc, f"${trueX // stride_unrolls}"):
                with For(cc, f"${trueX // portion_unroll_values}"):
                    for j in range(portion_unrolls):
                        for i in range(stride_unrolls):
                            offset_s = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                            offset_p = self.aligned_z(2 * trueX * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            offset_a = self.aligned_z(i * trueX * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            offset_q = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                            offset_broadcast = self.z(i * self.experiment.constants.dtype_size_bytes)

                            ymm_s = f"%{cc.get_register(register_set='simd', variable='s')}"
                            ymm_p = f"%{cc.get_register(register_set='simd', variable='p')}"
                            ymm_q = f"%{cc.get_register(register_set='simd', variable='q')}"
                            ymm_r = f"%{cc.get_register(register_set='simd', variable='r')}"
                            ymm_a = f"%{cc.get_register(register_set='simd', variable='a')}"

                            cc.add_statement("vmovaps", f"{offset_s}({S})", ymm_s)
                            cc.add_statement("vmovaps", f"{offset_p}({S})", ymm_p)
                            cc.add_statement("vmovaps", f"{offset_q}({stack_ptr})", ymm_q)
                            cc.add_statement("vbroadcastss", f"{offset_broadcast}({R})", ymm_r)
                            cc.add_statement("vmovaps", f"{offset_a}({D})", ymm_a)
                            cc.add_statement("vfmadd231ps", ymm_r, ymm_a, ymm_s)
                            cc.add_statement("vfmadd231ps", ymm_p, ymm_a, ymm_q)
                            cc.add_statement("vmovaps", ymm_q, f"{offset_q}({stack_ptr})")
                            cc.add_statement("vmovaps", ymm_s, f"{offset_s}({S})")

                            cc.unset_variable("s")
                            cc.unset_variable("p")
                            cc.unset_variable("q")
                            cc.unset_variable("r")
                            cc.unset_variable("a")

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", S)
                cc.add_statement("addq", f"${self.aligned(stride_unrolls * 8 * 4)}", stack_ptr)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", R)
                cc.add_statement("addq", f"${self.aligned(trueX * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueX * self.experiment.constants.dtype_size_bytes)}", S)
            cc.add_statement("subq", offset, D)



            if self.testing:
                cc.add_statement("subq", f"${self.aligned(trueX * self.experiment.constants.simd_vec_bytes)}", stack_ptr)
                with For(cc, f"${trueX}"):
                    q_vec_y = f"%{cc.get_register(register_set='simd', variable='q_vec')}"
                    q_vec_x = f"%{cc.get_variable('q_vec', size_column=2)}"
                    ancilla = f"%{cc.get_register(register_set='simd', variable='ancilla_vec', size_column=2)}"
        
                    cc.add_statement("vmovups", f"({stack_ptr})", q_vec_y)
                    cc.add_statement("vextractf128", "$1", q_vec_y, ancilla)
                    cc.add_statement("vaddps", q_vec_x, ancilla, q_vec_x)
                    cc.add_statement("vmovhlps", q_vec_x, q_vec_x, ancilla)
                    cc.add_statement("vaddps", q_vec_x, ancilla, q_vec_x)
                    cc.add_statement("vshufps", "$85", q_vec_x, q_vec_x, ancilla)
                    cc.add_statement("vaddss", ancilla, q_vec_x, q_vec_x)
                    cc.add_statement("vmovss", q_vec_x, f"({Q})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", stack_ptr)
                    cc.add_statement("addq", f"${self.experiment.constants.dtype_size_bytes}", Q)

                    cc.unset_variable("q_vec")
                    cc.unset_variable("ancilla_vec")

            cc.add_statement("movq", frame_ptr, stack_ptr)

            cc.unset_variable("D")
            cc.unset_variable("stack_ptr")
            cc.unset_variable("frame_ptr")
            cc.unset_variable("Q")
            cc.unset_variable("offset")
            cc.unset_variable("R")
            cc.unset_variable("S")