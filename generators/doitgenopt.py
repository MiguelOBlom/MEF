from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import DoitgenBaseGenerator

class DoitgenOptGenerator(DoitgenBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "doitgenopt")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        Q = configuration["Q"]
        R = configuration["R"]

        return 2 * R * Q * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return DoitgenOptGenerator.get_size_to_allocate(configuration)

    def build(self, configuration):
        P = configuration["P"]
        Q = configuration["Q"]
        R = configuration["R"]

        N = self.get_size_to_allocate_i(configuration)

        stride_unrolls_init = configuration["stride_unrolls_init"]
        portion_unrolls_init = configuration["portion_unrolls_init"]
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        stride_unrolls_write = configuration["stride_unrolls_write"]
        portion_unrolls_write = configuration["portion_unrolls_write"]
        unalignment_factor = configuration["unalignment_factor"]

        if stride_unrolls + portion_unrolls > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        trueP = self.get_true_N(P, [stride_unrolls_init, stride_unrolls, stride_unrolls_write], [portion_unrolls_init, portion_unrolls, portion_unrolls_write], unalignment_factor=unalignment_factor)

        if trueP is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueR = self.get_true_N(R, stride_unrolls_write, portion_unrolls_write)
        trueQ = Q

        if trueR != None and trueR > trueP:
            trueR = self.get_true_N(trueP, stride_unrolls_write, portion_unrolls_write)
        
        if trueR is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = self.get_size_to_allocate_i({"R": trueR, "Q": trueQ, "P": trueP})

        W_init = trueP // stride_unrolls_init
        W_write = trueP // stride_unrolls_write

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"], test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            Dorig = f"%{cc.get_register(variable='Dorig')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            pxp_offset = f"%{cc.get_register(variable='PxP')}"
            pxqxr_offset = f"%{cc.get_register(variable='PxQxR')}"
            C4 = f"%{cc.get_register(variable='C4')}"
            Sum = f"%{cc.get_register(variable='Sum')}"

            cc.add_statement("movq", f"${trueP * self.experiment.constants.dtype_size_bytes}", pxp_offset)
            cc.add_statement("movq", pxp_offset, pxqxr_offset)
            cc.add_statement("imul", f"${trueP}", pxp_offset)
            cc.add_statement("imul", f"${trueQ}", pxqxr_offset)
            cc.add_statement("imul", f"${trueR}", pxqxr_offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)
            self.aligned(trueP * trueQ * trueR * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, C4)
            cc.add_statement("addq", pxqxr_offset, C4)
            
            cc.add_statement("movq", C4, Sum)
            cc.add_statement("addq", pxp_offset, Sum)

            cc.add_statement("movq", D, Dorig)

            with For(cc, f"${trueR}"):
                with For(cc, f"${trueQ}"):
                    # Initialize Sum to zero
                    self.initialize_zero(cc, W_init, Sum, stride_unrolls_init, portion_unrolls_init)
                    
                    with For(cc, f"${trueP // stride_unrolls}"):
                        for i in range(stride_unrolls):
                            ymm_a = f"%{cc.get_register(register_set='simd', variable=f'a{i}')}"
                            offset_a = self.z(i * self.experiment.constants.dtype_size_bytes)
                            cc.add_statement("vbroadcastss", f"{offset_a}({D})", ymm_a)

                        with For(cc, f"${trueP // portion_unroll_values}"):
                            for j in range(portion_unrolls):
                                ymm_sum = f"%{cc.get_register(register_set='simd', variable=f'sum{j}')}"
                                offset_sum = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                                cc.add_statement("vmovaps", f"{offset_sum}({Sum})", ymm_sum)

                            for i in range(stride_unrolls):
                                for j in range(portion_unrolls):
                                    ymm_a = f"%{cc.get_variable(f'a{i}')}"
                                    ymm_sum = f"%{cc.get_variable(f'sum{j}')}"
                                    
                                    offset_c4 = self.aligned_z(trueP * i * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)

                                    cc.add_statement("vfmadd231ps", f"{offset_c4}({C4})", ymm_a, ymm_sum)

                            for j in range(portion_unrolls):
                                ymm_sum = cc.get_variable(f'sum{j}')
                                offset_sum = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                                cc.add_statement("vmovaps", f"%{ymm_sum}", f"{offset_sum}({Sum})")
                                cc.unset_variable(f'sum{j}')

                            cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", Sum)
                            cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes)}", C4)

                        cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", Sum)
                        cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls - 1) * self.experiment.constants.dtype_size_bytes)}", C4)
                        cc.add_statement("addq", f"${stride_unrolls * self.experiment.constants.dtype_size_bytes}", D)

                        for i in range(stride_unrolls):
                            cc.unset_variable(f"a{i}")

                    cc.add_statement("subq", f"${trueP * self.experiment.constants.dtype_size_bytes}", D)
                    cc.add_statement("subq", pxp_offset, C4)
                    
                    # Write Sum to D
                    self.writeback(cc, W_write, Sum, D, stride_unrolls_write, portion_unrolls_write)

                    cc.add_statement("addq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", Sum)
                    cc.add_statement("addq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", D)

            cc.add_statement("movq", Dorig, D)
            cc.unset_variable('D')
            cc.unset_variable('Dorig')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('PxP')
            cc.unset_variable('PxQxR')
            cc.unset_variable('C4')
            cc.unset_variable('Sum')
        return (trueN, N, {"trueR": trueR, "trueQ": trueQ, "trueP": trueP})
        


    def test(self, configuration, test_data_dir):
        P = configuration["P"]
        R = configuration["R"]
        Q = configuration["Q"]

        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        trueR = self.get_true_N(R, 1, 1)

        if not trueR is None and not trueP is None and trueR > trueP:
            trueR = self.get_true_N(trueP, 1, 1)

        if trueP is None or trueR is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueQ = Q

        if not trueR is None and not trueP is None and trueR > trueP:
            trueR = self.get_true_N(trueP, 1, 1)
        
        a = ((np.random.randint(100, size=(trueR, trueQ, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        c4 = ((np.random.randint(100, size=(trueP, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        sum = ((np.random.randint(100, size=(trueR, trueQ, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = self.get_remainder(configuration, {"P": trueP, "R": trueR, "Q": trueQ}).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, a, c4, sum, remainder)

        for r in range(trueR):
            for q in range(trueQ):
                for p in range(trueP):
                    sum[r][q][p] = 0

                for s in range(trueP):
                    for p in range(trueP):
                        sum[r][q][p] = sum[r][q][p] + a[r][q][s] * c4[s][p]
                
                for p in range(trueR):
                    a[r][q][p] = sum[r][q][p]


        self.write_test_output(test_data_dir, a, c4, sum, remainder)

        return test_data_dir
