import functools

class Constants:
    def __init__(self, entry_function,
                 dtype_size_bytes, simd_vec_bits,
                 nd_type,
                 warmup, repetitions, entries,
                 default_register_set, default_register_set_default_column,
                 simd_register_set, simd_register_set_default_column,
                 machine_config,
                 realpath,
                 construct_resource_path,
                 construct_result_path,
                 test_input_filename, test_output_filename):
        self.entry_function = entry_function
        self.dtype_size_bytes = dtype_size_bytes
        self.warmup = warmup
        self.repetitions = repetitions
        self.entries = entries
        self.simd_vec_bits = simd_vec_bits
        self.simd_vec_bytes = simd_vec_bits // 8
        self.simd_vec_values = self.simd_vec_bytes // dtype_size_bytes
        self.nd_type = nd_type
        self.default_register_set = default_register_set
        self.default_register_set_default_column = default_register_set_default_column
        self.simd_register_set = simd_register_set
        self.simd_register_set_default_column = simd_register_set_default_column
        self.machine_config = machine_config
        self.realpath = realpath
        self.construct_resource_path = construct_resource_path
        self.construct_result_path = construct_result_path
        self.test_input_filename = test_input_filename
        self.test_output_filename = test_output_filename
        self.n_default_regs = len(default_register_set)
        self.n_simd_regs = len(simd_register_set)

    def get_all_registers(self, simd=False):
        # Get the names of all registers in the specified set
        registers = []

        register_set = self.simd_register_set if simd else self.default_register_set

        for register_description in register_set:
            registers.append(register_description[0][self.simd_register_set_default_column if simd else self.default_register_set_default_column])
        
        return registers