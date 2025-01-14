from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .base import GemverOuterBaseGenerator

class GemverOptGenerator(GemverOuterBaseGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "gemveropt")
        self.testing = testing

    @staticmethod
    def get_size_to_allocate(configuration):
        P = configuration["P"]
        return 8 * P + P * P

    def get_size_to_allocate_i(self, configuration):
        return GemverOptGenerator.get_size_to_allocate(configuration)


    def build(self, configuration):
        P = configuration["P"]

        N = self.get_size_to_allocate_i(configuration)

        mode = configuration["mode"]

        stride_unrolls_outer = configuration["stride_unrolls_outer"]
        portion_unrolls_outer = configuration["portion_unrolls_outer"]
        portion_unroll_values_outer = portion_unrolls_outer * self.experiment.constants.simd_vec_values
        portion_unroll_bytes_outer = portion_unrolls_outer * self.experiment.constants.simd_vec_bytes

        stride_unrolls_mxvt = configuration["stride_unrolls_mxvt"]
        portion_unrolls_mxvt= configuration["portion_unrolls_mxvt"]
        portion_unroll_values_mxvt = portion_unrolls_mxvt * self.experiment.constants.simd_vec_values
        portion_unroll_bytes_mxvt = portion_unrolls_mxvt * self.experiment.constants.simd_vec_bytes

        stride_unrolls_sum = configuration["stride_unrolls_sum"]
        portion_unrolls_sum = configuration["portion_unrolls_sum"]
        portion_unroll_values_sum = portion_unrolls_sum * self.experiment.constants.simd_vec_values
        portion_unroll_bytes_sum = portion_unrolls_sum * self.experiment.constants.simd_vec_bytes

        stride_unrolls_mxv = configuration["stride_unrolls_mxv"]
        portion_unrolls_mxv = configuration["portion_unrolls_mxv"]
        portion_unroll_values_mxv = portion_unrolls_mxv * self.experiment.constants.simd_vec_values
        portion_unroll_bytes_mxv = portion_unrolls_mxv * self.experiment.constants.simd_vec_bytes

        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, [stride_unrolls_outer, stride_unrolls_mxvt, stride_unrolls_sum, stride_unrolls_mxv], [portion_unrolls_outer, portion_unrolls_mxvt, portion_unrolls_sum, portion_unrolls_mxv], unalignment_factor=unalignment_factor)
        
        if trueP is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = self.get_size_to_allocate_i({"P": trueP})

        if stride_unrolls_outer * 2 + 3 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        if stride_unrolls_mxvt + 2 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return
        
        if stride_unrolls_mxv + 2 > self.experiment.constants.n_simd_regs:
            Logger.note(f"Not generating for more than {self.experiment.constants.n_simd_regs} registers!")
            return

        with CodeContext(self, mode, 1, N, trueN, suffix=configuration["suffix"], test_function_configuration=(self.test, configuration) if self.testing else None) as cc:
            D = f"%{cc.set_variable('rdi', variable='D')}"
            stack_ptr = f"%{cc.set_variable('rsp', variable='stack_ptr')}"
            offset = f"%{cc.get_register(variable='offset')}"

            # Outer

            U1 = f"%{cc.get_register(variable='U1')}"
            V1 = f"%{cc.get_register(variable='V1')}"

            cc.add_statement("movq", f"${trueP}", offset)
            cc.add_statement("imul", f"${trueP * self.experiment.constants.dtype_size_bytes}", offset)
            self.aligned(trueP * trueP * self.experiment.constants.dtype_size_bytes)

            cc.add_statement("movq", D, U1)
            cc.add_statement("addq", offset, U1)
            
            cc.add_statement("leaq", f"{self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}({U1})", V1)

            with For(cc, f"${trueP // stride_unrolls_outer}"):
                for i in range(stride_unrolls_outer):
                    offset_u1 = self.z(i * self.experiment.constants.dtype_size_bytes)
                    offset_u2 = self.z(2 * trueP * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.dtype_size_bytes)

                    ymm_u1 = f"%{cc.get_register(register_set='simd', variable=f'u1{i}')}"
                    ymm_u2 = f"%{cc.get_register(register_set='simd', variable=f'u2{i}')}"

                    cc.add_statement("vbroadcastss", f"{offset_u1}({U1})", ymm_u1)
                    cc.add_statement("vbroadcastss", f"{offset_u2}({U1})", ymm_u2)

                with For(cc, f"${trueP // portion_unroll_values_outer}"):
                    for j in range(portion_unrolls_outer):
                        offset_v1 = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                        offset_v2 = self.aligned_z(2 * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                
                        ymm_v1 = cc.get_register(register_set='simd', variable=f'v1{j}')
                        ymm_v2 = cc.get_register(register_set='simd', variable=f'v2{j}')
                        
                        cc.add_statement("vmovaps", f"{offset_v1}({V1})", f"%{ymm_v1}")
                        cc.add_statement("vmovaps", f"{offset_v2}({V1})", f"%{ymm_v2}")

                        for i in range(stride_unrolls_outer):
                            ymm_u1 = f"%{cc.get_variable(f'u1{i}')}"
                            ymm_u2 = f"%{cc.get_variable(f'u2{i}')}"
                            ymm_a = f"%{cc.get_register(register_set='simd', variable=f'a')}"

                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)

                            cc.add_statement("vmovaps", f"{offset_a}({D})", ymm_a)
                            cc.add_statement("vfmadd231ps", f"%{ymm_v1}", ymm_u1, ymm_a)
                            cc.add_statement("vfmadd231ps", f"%{ymm_v2}", ymm_u2, ymm_a)
                            cc.add_statement("vmovaps", ymm_a, f"{offset_a}({D})")

                            cc.unset_variable('a')

                        cc.unset_variable(f'v1{j}')
                        cc.unset_variable(f'v2{j}')

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_outer)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_outer)}", V1)

                for i in range(stride_unrolls_outer):
                    cc.unset_variable(f'u1{i}')
                    cc.unset_variable(f'u2{i}')
                
                cc.add_statement("addq", f"${self.aligned((stride_unrolls_outer - 1) * trueP * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("addq", f"${stride_unrolls_outer * self.experiment.constants.dtype_size_bytes}", U1)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", V1)
            
            cc.add_statement("subq", offset, D)
            cc.unset_variable('U1')
            cc.unset_variable('V1')

            # MxV T
            X = f"%{cc.get_register(variable='X')}"
            Y = f"%{cc.get_register(variable='Y')}"
            ymm_beta = f"%{cc.get_register(register_set='simd', variable='beta')}"

            cc.add_statement("leaq", f"{self.aligned_z(5 * trueP * self.experiment.constants.dtype_size_bytes)}({D})", X)
            cc.add_statement("addq", offset, X)
            cc.add_statement("leaq", f"{self.aligned_z(trueP * self.experiment.constants.dtype_size_bytes)}({X})", Y)

            cc.add_statement("movl", "$3213675725", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_beta)

            with For(cc, f"${trueP // stride_unrolls_mxvt}"):
                for j in range(stride_unrolls_mxvt):
                    ymm_y = f"%{cc.get_register(register_set='simd', variable=f'y{j}')}"
                    offset_y = self.z(j * self.experiment.constants.dtype_size_bytes)
                    cc.add_statement("vbroadcastss", f"{offset_y}({Y})", ymm_y)
                    cc.add_statement("vmulps", ymm_y, ymm_beta, ymm_y)

                with For(cc, f"${trueP // portion_unroll_values_mxvt}"):
                    for i in range(portion_unrolls_mxvt):
                        offset_x = self.aligned_z(i * self.experiment.constants.simd_vec_bytes)
                        ymm_x = cc.get_register(register_set='simd', variable=f'x{j}')

                        cc.add_statement("vmovaps", f"{offset_x}({X})", f"%{ymm_x}")

                        for j in range(stride_unrolls_mxvt):
                            ymm_y = f"%{cc.get_variable(f'y{j}')}"
                            offset_a = self.aligned_z(j * trueP * self.experiment.constants.dtype_size_bytes + i * self.experiment.constants.simd_vec_bytes)
                          
                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", ymm_y, f"%{ymm_x}")

                        cc.add_statement("vmovaps", f"%{ymm_x}", f"{offset_x}({X})")

                        cc.unset_variable(f'x{j}')

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_mxvt)}", X)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_mxvt)}", D)

                for j in range(stride_unrolls_mxvt):
                    cc.unset_variable(f'y{j}')

                cc.add_statement("addq", f"${self.aligned((stride_unrolls_mxvt - 1) * trueP * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", X)
                cc.add_statement("addq", f"${stride_unrolls_mxvt * self.experiment.constants.dtype_size_bytes}", Y)       

            cc.add_statement("subq", offset, D)


            cc.unset_variable('Y')
            cc.unset_variable('beta')

            # Sum
            with For (cc, f"${trueP // (portion_unroll_values_sum * stride_unrolls_sum)}"):
                for j in range(portion_unrolls_sum):
                    for i in range(stride_unrolls_sum):
                        offset_x = self.aligned_z(i * (trueP // stride_unrolls_sum) * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                        offset_z = self.aligned_z(i * (trueP // stride_unrolls_sum) * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes + 2 * trueP * self.experiment.constants.dtype_size_bytes)
                        
                        ymm = f"%{cc.get_register(register_set='simd', variable='vec')}"

                        cc.add_statement("vmovaps", f"{offset_x}({X})", ymm)
                        cc.add_statement("vaddps", f"{offset_z}({X})", ymm, ymm)
                        cc.add_statement("vmovaps", ymm, f"{offset_x}({X})")
                        
                        cc.unset_variable("vec")

                cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_sum)}", X)

            # MxV
            W = f"%{cc.get_register(variable='W')}"
            ymm_alpha = f"%{cc.get_register(register_set='simd', variable='alpha')}"

            cc.add_statement("subq", f"${self.aligned((trueP // stride_unrolls_sum) * self.experiment.constants.dtype_size_bytes)}", X)
            cc.add_statement("leaq", f"{self.aligned_z(-trueP * self.experiment.constants.dtype_size_bytes)}({X})", W)

            cc.add_statement("movl", "$1083179008", f"-8({stack_ptr})")
            cc.add_statement("vbroadcastss", f"-8({stack_ptr})", ymm_alpha)
           
            with For(cc, f"${trueP // stride_unrolls_mxv}"):
                for i in range(stride_unrolls_mxv):
                    cc.get_register(register_set="simd", variable=f"w{i}")
                first = f"%{cc.get_variable('w0')}"

                cc.add_statement("vxorps", first, first, first)
                for i in range(1, stride_unrolls_mxv):
                    src = f"%{cc.get_variable(f'w{i-1}')}"
                    dst = f"%{cc.get_variable(f'w{i}')}"
                    cc.add_statement("vmovaps", src, dst)

                with For(cc, f"${trueP // portion_unroll_values_mxv}"):
                    for j in range(portion_unrolls_mxv):
                        offset_x = self.aligned_z(j * self.experiment.constants.simd_vec_bytes)
                        ymm_x = cc.get_register(register_set='simd', variable=f'x{j}')

                        cc.add_statement("vmulps", f"{offset_x}({X})", ymm_alpha, f"%{ymm_x}")

                        for i in range(stride_unrolls_mxv):
                            offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                            ymm_w = f"%{cc.get_variable(f'w{i}')}"

                            cc.add_statement("vfmadd231ps", f"{offset_a}({D})", f"%{ymm_x}", ymm_w)

                        cc.unset_variable(f'x{j}')

                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_mxv)}", D)
                    cc.add_statement("addq", f"${self.aligned(portion_unroll_bytes_mxv)}", X)

                for i in range(stride_unrolls_mxv):
                    offset_a = self.aligned_z(i * trueP * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                    ymm_w = cc.get_variable(f'w{i}')
                    xmm_w = cc.get_variable(f'w{i}', size_column=2)
                    ancilla = cc.get_register(register_set='simd', variable='ancilla', size_column=2)

                    offset_w = self.z(i * self.experiment.constants.dtype_size_bytes)

                    cc.add_statement("vextractf128", "$0x1", f"%{ymm_w}", f"%{ancilla}")
                    cc.add_statement("vaddps", f"%{xmm_w}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vhaddps", f"%{ancilla}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vhaddps", f"%{ancilla}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vbroadcastss", f"{offset_w}({W})", f"%{xmm_w}")
                    cc.add_statement("vaddss", f"%{xmm_w}", f"%{ancilla}", f"%{ancilla}")
                    cc.add_statement("vmovss", f"%{ancilla}", f"{offset_w}({W})")

                    cc.unset_variable(f'w{i}')
                    cc.unset_variable('ancilla')


                cc.add_statement("addq", f"${self.aligned(trueP * (stride_unrolls_mxv - 1) * self.experiment.constants.dtype_size_bytes)}", D)
                cc.add_statement("subq", f"${self.aligned(trueP * self.experiment.constants.dtype_size_bytes)}", X)
                cc.add_statement("addq", f"${stride_unrolls_mxv * self.experiment.constants.dtype_size_bytes}", W)
                
            cc.add_statement("subq", offset, D)

            cc.unset_variable('D')
            cc.unset_variable('X')
            cc.unset_variable('W')
            cc.unset_variable('stack_ptr')
            cc.unset_variable('offset')
            cc.unset_variable('alpha')
        return (trueN, N, {"trueP": trueP})


    def test(self, configuration, test_data_dir):
        P = configuration["P"]

        mode = configuration["mode"]

        stride_unrolls_outer = configuration["stride_unrolls_outer"]
        portion_unrolls_outer = configuration["portion_unrolls_outer"]
        stride_unrolls_mxvt = configuration["stride_unrolls_mxvt"]
        portion_unrolls_mxvt= configuration["portion_unrolls_mxvt"]
        stride_unrolls_sum = configuration["stride_unrolls_sum"]
        portion_unrolls_sum = configuration["portion_unrolls_sum"]
        stride_unrolls_mxv = configuration["stride_unrolls_mxv"]
        portion_unrolls_mxv = configuration["portion_unrolls_mxv"]

        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, [stride_unrolls_outer, stride_unrolls_mxvt, stride_unrolls_sum, stride_unrolls_mxv], [portion_unrolls_outer, portion_unrolls_mxvt, portion_unrolls_sum, portion_unrolls_mxv], unalignment_factor=unalignment_factor)
        

        a = ((np.random.randint(100, size=(trueP, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        u1 = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        v1 = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        u2 = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        v2 = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        w = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        x = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        y = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        z = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = (self.get_remainder(configuration, {"P": trueP})).astype(self.experiment.constants.nd_type)
        
        self.write_test_input(test_data_dir, a, u1, v1, u2, v2, w, x, y, z, remainder)

        for i in range(trueP):
            for j in range(trueP):
                a[i][j] += u1[i] * v1[j] + u2[i] * v2[j]

        for i in range(trueP):
            for j in range(trueP):
                x[i] = x[i] + -1.1 * a[j][i] * y[j]

        for i in range(trueP):
            x[i] += z[i]
        
        for i in range(trueP):
            for j in range(trueP):
                w[i] = w[i] + 4.5 * a[i][j] * x[j]

        self.write_test_output(test_data_dir, a, u1, v1, u2, v2, w, x, y, z, remainder)

        return test_data_dir