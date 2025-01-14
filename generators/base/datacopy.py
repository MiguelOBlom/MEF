from classes import Generator, CodeContext, For, Logger
import numpy as np
import os

class DataCopyGenerator(Generator):
    def __init__(self, experiment, kernel_name, testing=False):
        super().__init__(experiment, kernel_name, testing=testing)

    def build(self, configuration):
        N = configuration["N"] 
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        portion_unroll_bytes = portion_unrolls * self.experiment.constants.simd_vec_bytes
        unalignment_factor = configuration["unalignment_factor"]
        
        I = N // 2
        trueI = self.get_true_N(I, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor, page_size_bytes=None)

        if trueI is None:
            Logger.warn(f"Cannot generate for {self.kernel_name}")
            return

        trueN = trueI * 2
        W = trueI // stride_unrolls

        with CodeContext(self, stride_unrolls, portion_unrolls, N, trueN, test_function_configuration=(self.test, configuration) if self.testing else None, suffix=configuration["suffix"]) as cc:
            cc.set_variable('rdi', 'D')
            cc.set_variable('rsp', 'stack_ptr')
            cc.add_statement(f"movq", f"%{cc.get_variable('D')}", f"%{cc.get_register(variable='I')}")
            cc.add_statement("leaq", f"{self.aligned(trueI * self.experiment.constants.dtype_size_bytes)}(%{cc.get_variable('I')})", f"%{cc.get_register(variable='O')}")

            with For (cc, f"${W // portion_unroll_values}"):
                for i in range(stride_unrolls):
                    for j in range(portion_unrolls):
                        offset = self.aligned_z(i * W * self.experiment.constants.dtype_size_bytes + j * self.experiment.constants.simd_vec_bytes)
                        cc.add_statement(*self.build_main_operation_load(f"%{cc.get_register(register_set='simd', variable='vec')}", f"{offset}(%{cc.get_variable('I')})"))
                        cc.add_statement(*self.build_main_operation_store(f"%{cc.get_variable('vec')}", f"{offset}(%{cc.get_variable('O')})"))
                        cc.unset_variable('vec')
                cc.add_statement(f"addq", f"${self.aligned(portion_unroll_bytes)}", f"%{cc.get_variable('I')}")
                cc.add_statement(f"addq", f"${self.aligned(portion_unroll_bytes)}", f"%{cc.get_variable('O')}")

            cc.unset_variable('D')
            cc.unset_variable('I')
            cc.unset_variable('O')
            cc.unset_variable('stack_ptr')


    def test(self, configuration, test_data_dir):
        N = configuration["N"] 
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        unalignment_factor = configuration["unalignment_factor"]
        
        I = N // 2
        trueI = self.get_true_N(I, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor, page_size_bytes=None)
        W = trueI // stride_unrolls

        m = ((np.random.randint(100, size=(N)) / 10) - 5).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, m)

        for x in range(0, W // portion_unroll_values):
            for i in range(stride_unrolls):
                for j in range(portion_unrolls):
                    for k in range(self.experiment.constants.simd_vec_values):
                        offset = i * W +  j * self.experiment.constants.simd_vec_values + k
                        m[trueI + x * portion_unroll_values + offset]  = m[x * portion_unroll_values + offset]
                    
        self.write_test_output(test_data_dir, m)

        return test_data_dir

    def build_main_operation(self, vec, mem):
        Logger.fail("build_main_operation not implemented in DataMovementGenerator, inherrit from this class.")
