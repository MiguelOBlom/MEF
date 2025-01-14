from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import ComputeBaseGenerator

class Convolution3x3Generator(ComputeBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "convolution3x3")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        X = configuration["X"]
        return 2 * X * X + 16

    def get_size_to_allocate_i(self, configuration):
        return Convolution3x3Generator.get_size_to_allocate(configuration)

    def build(self, configuration):
        X = configuration["X"]
        N = self.get_size_to_allocate_i(configuration)
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]
        
        trueX = self.get_true_N(X - 2, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        
        if trueX is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueX += 2

        trueN = self.get_size_to_allocate_i({"X": trueX})

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, suffix=configuration["suffix"] if "suffix" in configuration.keys() else "",  test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"
            O = f"%{cc.get_register(variable='output')}"
            I = f"%{cc.get_register(variable='input')}"
            
            RL = f"%{cc.get_register(register_set='simd', variable='right_lower')}"
            CL = f"%{cc.get_register(register_set='simd', variable='center_lower')}"
            LL = f"%{cc.get_register(register_set='simd', variable='left_lower')}"
            RC = f"%{cc.get_register(register_set='simd', variable='right_center')}"
            CC = f"%{cc.get_register(register_set='simd', variable='center_center')}"
            LC = f"%{cc.get_register(register_set='simd', variable='left_center')}"
            RU = f"%{cc.get_register(register_set='simd', variable='right_upper')}"
            CU = f"%{cc.get_register(register_set='simd', variable='center_upper')}"
            LU = f"%{cc.get_register(register_set='simd', variable='left_upper')}"

            cc.add_statement("movq", f"${trueX}", offset)
            cc.add_statement("imul", f"${trueX * self.experiment.constants.dtype_size_bytes}", offset)

            cc.add_statement("leaq", f"{16 * self.experiment.constants.dtype_size_bytes}({D})", I)
            cc.add_statement("movq", I, O)
            cc.add_statement("addq", offset, O)

            cc.add_statement("vbroadcastss", f"({D})", LU)
            cc.add_statement("vbroadcastss", f"{self.experiment.constants.dtype_size_bytes}({D})", CU)
            cc.add_statement("vbroadcastss", f"{2 * self.experiment.constants.dtype_size_bytes}({D})", RU)
            cc.add_statement("vbroadcastss", f"{3 * self.experiment.constants.dtype_size_bytes}({D})", LC)
            cc.add_statement("vbroadcastss", f"{4 * self.experiment.constants.dtype_size_bytes}({D})", CC)
            cc.add_statement("vbroadcastss", f"{5 * self.experiment.constants.dtype_size_bytes}({D})", RC)
            cc.add_statement("vbroadcastss", f"{6 * self.experiment.constants.dtype_size_bytes}({D})", LL)
            cc.add_statement("vbroadcastss", f"{7 * self.experiment.constants.dtype_size_bytes}({D})", CL)
            cc.add_statement("vbroadcastss", f"{8 * self.experiment.constants.dtype_size_bytes}({D})", RL)

            with For(cc, f"${(trueX - 2) // stride_unrolls}"):
                with For(cc, f"${(trueX - 2) // portion_unroll_values}"):
                    for i in range(portion_unrolls):
                        for j in range(stride_unrolls):
                            out_vec = f"%{cc.get_register(register_set='simd', variable='out')}"

                            offset_lu = self.z(((j + 0) * trueX + i * self.experiment.constants.simd_vec_values + 0) * self.experiment.constants.dtype_size_bytes)
                            offset_cu = self.z(((j + 0) * trueX + i * self.experiment.constants.simd_vec_values + 1) * self.experiment.constants.dtype_size_bytes)
                            offset_ru = self.z(((j + 0) * trueX + i * self.experiment.constants.simd_vec_values + 2) * self.experiment.constants.dtype_size_bytes)
                            offset_lc = self.z(((j + 1) * trueX + i * self.experiment.constants.simd_vec_values + 0) * self.experiment.constants.dtype_size_bytes)
                            offset_cc = self.z(((j + 1) * trueX + i * self.experiment.constants.simd_vec_values + 1) * self.experiment.constants.dtype_size_bytes)
                            offset_rc = self.z(((j + 1) * trueX + i * self.experiment.constants.simd_vec_values + 2) * self.experiment.constants.dtype_size_bytes)
                            offset_ld = self.z(((j + 2) * trueX + i * self.experiment.constants.simd_vec_values + 0) * self.experiment.constants.dtype_size_bytes)
                            offset_cd = self.z(((j + 2) * trueX + i * self.experiment.constants.simd_vec_values + 1) * self.experiment.constants.dtype_size_bytes)
                            offset_rd = self.z(((j + 2) * trueX + i * self.experiment.constants.simd_vec_values + 2) * self.experiment.constants.dtype_size_bytes)

                            cc.add_statement("vmulps", f"{offset_lu}({I})", LU, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_cu}({I})", CU, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_ru}({I})", RU, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_lc}({I})", LC, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_cc}({I})", CC, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_rc}({I})", RC, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_ld}({I})", LL, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_cd}({I})", CL, out_vec)
                            cc.add_statement("vfmadd231ps", f"{offset_rd}({I})", RL, out_vec)
                            cc.add_statement("vmovups", out_vec, f"{offset_cc}({O})")

                            cc.unset_variable("out")

                    cc.add_statement("addq", f"${portion_unroll_bytes}", I)
                    cc.add_statement("addq", f"${portion_unroll_bytes}", O)
                cc.add_statement("addq", f"${((stride_unrolls - 1) * trueX + 2) * self.experiment.constants.dtype_size_bytes}", I)
                cc.add_statement("addq", f"${((stride_unrolls - 1) * trueX + 2) * self.experiment.constants.dtype_size_bytes}", O)

            cc.unset_variable('right_lower')
            cc.unset_variable('center_lower')
            cc.unset_variable('left_lower')
            cc.unset_variable('right_center')
            cc.unset_variable('center_center')
            cc.unset_variable('left_center')
            cc.unset_variable('right_upper')
            cc.unset_variable('center_upper')
            cc.unset_variable('left_upper')

            cc.unset_variable('D')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('output')
            cc.unset_variable('input')
        return (trueN, N, {"trueX": trueX, "prebuild_side": trueX})


    def test(self, configuration, test_data_dir):
        X = configuration["X"]
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X - 2, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor) + 2

        c = ((np.random.randint(100, size=(16)) / 10) - 5).astype(self.experiment.constants.nd_type)
        a = ((np.random.randint(100, size=(trueX, trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        b = ((np.random.randint(100, size=(trueX, trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = self.get_remainder(configuration, {"X": trueX}).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, c, a, b, remainder)

        for i in range(1, trueX - 1):
            for j in range(1, trueX - 1):
                b[i,j] = a[i - 1, j - 1] * c[0]
                b[i,j] += a[i - 1, j + 0] * c[1]
                b[i,j] += a[i - 1, j + 1] * c[2]
                b[i,j] += a[i + 0, j - 1] * c[3]
                b[i,j] += a[i + 0, j + 0] * c[4]
                b[i,j] += a[i + 0, j + 1] * c[5]
                b[i,j] += a[i + 1, j - 1] * c[6]
                b[i,j] += a[i + 1, j + 0] * c[7]
                b[i,j] += a[i + 1, j + 1] * c[8]
        
        self.write_test_output(test_data_dir, c, a, b, remainder)

        return test_data_dir
    
    def test_halide(self, configuration, test_data_dir):
        N = configuration[1]
        trueX = configuration[2]["trueX"]
        Nremainder = N - self.get_size_to_allocate_i({"X": trueX})

        c = ((np.random.randint(100, size=(16)) / 10) - 5).astype(self.experiment.constants.nd_type)
        a = ((np.random.randint(100, size=(trueX, trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        b = ((np.random.randint(100, size=(trueX - 2, trueX - 2)) / 10) - 5).astype(self.experiment.constants.nd_type)
        p = ((np.random.randint(100, size=(4 * trueX - 4)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = ((np.random.randint(100, size=(Nremainder)) / 10) - 5).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, c, a, b, p, remainder)

        for i in range(1, trueX - 1):
            for j in range(1, trueX - 1):
                b[i - 1, j - 1] = a[i - 1, j - 1] * c[0]
                b[i - 1, j - 1] += a[i - 1, j + 0] * c[1]
                b[i - 1, j - 1] += a[i - 1, j + 1] * c[2]
                b[i - 1, j - 1] += a[i + 0, j - 1] * c[3]
                b[i - 1, j - 1] += a[i + 0, j + 0] * c[4]
                b[i - 1, j - 1] += a[i + 0, j + 1] * c[5]
                b[i - 1, j - 1] += a[i + 1, j - 1] * c[6]
                b[i - 1, j - 1] += a[i + 1, j + 0] * c[7]
                b[i - 1, j - 1] += a[i + 1, j + 1] * c[8]

        self.write_test_output(test_data_dir, c, a, b, p, remainder)

        return test_data_dir

