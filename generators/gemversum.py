from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverSumBaseGenerator

class GemverSumGenerator(GemverSumBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemversum")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P
    
    def get_size_to_allocate_i(self, configuration):
        return GemverSumGenerator.get_size_to_allocate(configuration)

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

        W = trueP // stride_unrolls

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"] if "suffix" in configuration.keys() else "", test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            X = f"%{cc.get_register(variable='X')}"
            Z = f"%{cc.get_register(variable='Z')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * 4)

            cc.add_statement("movq", D, X)
            cc.add_statement("movq", D, Z)
            cc.add_statement("addq", offset, Z)
            
            with For (cc, f"${W // portion_unroll_values}"):
                for j in range(portion_unrolls):
                    for i in range(stride_unrolls):
                        offset_addr = self.aligned_z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                        
                        ymm = f"%{cc.get_register(register_set='simd', variable='vec')}"

                        cc.add_statement("vmovaps", f"{offset_addr}({X})", ymm)
                        cc.add_statement("vaddps", f"{offset_addr}({Z})", ymm, ymm)
                        cc.add_statement("vmovaps", ymm, f"{offset_addr}({X})")
                        
                        cc.unset_variable("vec")

                cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", X)
                cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", Z)
            
            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('X')
            cc.unset_variable('Z')
        return (trueN, N, {"trueP": trueP})