from .base import DoitgenBaseGenerator
from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class DoitgenGenerator(DoitgenBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "doitgen")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 2 * P + P * P
    
    def get_size_to_allocate_i(self, configuration):
        return DoitgenGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        P = configuration["P"]
        R = configuration["R"]

        N = self.get_size_to_allocate_i(configuration)

        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        trueR = self.get_true_N(R, 1, 1)

        if not trueR is None and not trueP is None and trueR > trueP:
            trueR = self.get_true_N(trueP, 1, 1)

        if trueP is None or trueR is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return
        
        trueN = self.get_size_to_allocate_i({"P": trueP})

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            C4 = f"%{cc.get_register(variable='C4')}"
            Sum = f"%{cc.get_register(variable='Sum')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("leaq", f"{self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}({D})", C4)
            cc.add_statement("movq", C4, Sum)
            cc.add_statement("addq", offset, Sum)

            if self.testing:
                zero_vec = f"%{cc.get_register(register_set='simd', variable='zero_vec')}"
                
                cc.add_statement("vxorps", zero_vec, zero_vec, zero_vec)
                with For (cc, f"${trueP // self.experiment.constants.simd_vec_values}"):
                    cc.add_statement("vmovntdq", zero_vec, f"({Sum})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", Sum)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", Sum)

                cc.unset_variable("zero_vec")
                
            with For(cc, f"${trueP // stride_unrolls}"):
                with For(cc, f"${trueP // portion_unroll_values}"):
                    for i in range(portion_unrolls):
                        for j in range(stride_unrolls):
                            ymm_sum = f"%{cc.get_register(register_set='simd', variable='sum')}"
                            ymm_a = f"%{cc.get_register(register_set='simd', variable='a')}"

                            offset_sum = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                            offset_a = self.z(j * self.experiment.constants.dtype_size_bytes)
                            offset_c4 = self.aligned_z(i * self.experiment.constants.simd_vec_bytes + j * trueP * self.experiment.constants.dtype_size_bytes)

                            cc.add_statement("vmovaps", f"{offset_sum}({Sum})", ymm_sum)
                            cc.add_statement("vbroadcastss", f"{offset_a}({D})", ymm_a)
                            cc.add_statement("vfmadd231ps", f"{offset_c4}({C4})", ymm_a, ymm_sum)
                            cc.add_statement("vmovaps", ymm_sum, f"{offset_sum}({Sum})")

                            cc.unset_variable("sum")
                            cc.unset_variable("a")

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", C4)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", Sum)
                
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", Sum)
                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", C4)
                cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", D)
      
            cc.add_statement("subq", f"${trueP * self.experiment.constants.dtype_size_bytes}", D)
        
            if self.testing:
                with For (cc, f"${trueR // self.experiment.constants.simd_vec_values}"):
                    ymm_sum = f"%{cc.get_register(register_set='simd', variable='sum')}"
                    
                    cc.add_statement("vmovaps", f"({Sum})", ymm_sum)
                    cc.add_statement("vmovntdq", ymm_sum, f"({D})")
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", Sum)
                    cc.add_statement("addq", f"${self.experiment.constants.simd_vec_bytes}", D)
                    
                    cc.unset_variable("sum")

            cc.add_statement("subq", f"${trueR * self.experiment.constants.dtype_size_bytes}", D)

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('C4')
            cc.unset_variable('Sum')

    def test(self, configuration, test_data_dir):
        P = configuration["P"]
        R = configuration["R"]
        Q = configuration["Q"]

        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        trueR = self.get_true_N(R, 1, 1)
        trueQ = Q

        if not trueR is None and not trueP is None and trueR > trueP:
            trueR = self.get_true_N(trueP, 1, 1)
        
        a = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        c4 = ((np.random.randint(100, size=(trueP, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        sum = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = self.get_remainder(configuration, {"P": trueP, "R": trueR, "Q": trueQ}).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, a, c4, sum, remainder)

        for q in range(trueQ):
            for p in range(trueP):
                sum[p] = 0

                for s in range(trueP):
                    sum[p] = sum[p] + a[s] * c4[s][p]
                
            for p in range(trueR):
                a[p] = sum[p]

        self.write_test_output(test_data_dir, a, c4, sum, remainder)

        return test_data_dir
