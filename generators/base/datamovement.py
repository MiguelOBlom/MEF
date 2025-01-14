from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class DataMovementGenerator(Generator):
    def __init__(self, experiment, kernel_name, testing=False, unaligned=False, grouped=True, initzero=False):
        super().__init__(experiment, kernel_name, testing=testing)
        self.unaligned = unaligned
        self.grouped = grouped
        self.initzero = initzero

    def build(self, configuration):
        N = configuration["N"] 
        if self.unaligned:
            N += 4
            self.compiler.dmacro["N"] += 4
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]

        trueN = self.get_true_N(N, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor, page_size_bytes=None)

        if trueN is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        W = trueN // stride_unrolls

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None, suffix=configuration["suffix"]) as cc:
            cc.set_variable('rdi', 'D')
            cc.set_variable('rsp', 'stack_ptr')

            if self.initzero:
                registers = self.experiment.constants.get_all_registers(simd=True)
                first = f"%{registers[0]}"
                cc.add_statement("vxorps", first, first, first)
                for src, dst in zip(registers[:-1], registers[1:]):
                    cc.add_statement("vmovaps", f"%{src}", f"%{dst}")

            cc.add_statement(f"movq", f"%{cc.get_variable('D')}", f"%{cc.get_register(variable='Dp')}")
            with For (cc, f"${W // portion_unroll_values}"):
                indices = []

                if self.grouped:
                    for i in range(stride_unrolls):
                        for j in range(portion_unrolls):
                            indices.append((i, j))
                else:
                    for j in range(portion_unrolls):
                        for i in range(stride_unrolls):
                            indices.append((i, j))
                
                for i, j in indices:
                    if self.unaligned:
                        offset = self.z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes + 4)
                    else:
                        offset = self.aligned_z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)

                    cc.add_statement(*self.build_main_operation(f"%{cc.get_register(register_set='simd', variable='vec')}", f"{offset}(%{cc.get_variable('Dp')})"))
                    cc.unset_variable('vec')
                cc.add_statement(f"addq", f"${self.aligned(portion_unroll_bytes)}", f"%{cc.get_variable('Dp')}")

            cc.unset_variable("D")
            cc.unset_variable("Dp")
            cc.unset_variable("stack_ptr")

    def test(self, configuration, test_data_dir):
        N = configuration["N"] 
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]

        trueN = self.get_true_N(N, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor, page_size_bytes=None)

        W = trueN // stride_unrolls

        m = ((np.random.randint(100, size=(N)) / 10) - 5).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, m)

        for k in range (W // portion_unroll_values):
            for i in range(stride_unrolls):
                for j in range(portion_unroll_values):
                    m[k * portion_unroll_values + i * W + j] = 0
        
        self.write_test_output(test_data_dir, m)

        return test_data_dir

    def build_main_operation(self, vec, mem):
        Logger.fail("build_main_operation not implemented in DataMovementGenerator, inherrit from this class.")

    
