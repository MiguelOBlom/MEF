import re
import os
import math
import multiprocessing
import traceback
import functools
import numpy as np
from .logger import Logger

class Generator:
    def __init__(self, experiment, kernel_name, testing=False):
        self.experiment = experiment
        self.kernel_name = kernel_name
        self.commands = []
        self.testing = testing
        self.test_functions = []

    def lcm(self, l):
        lcm = 1
        for n in l:
            lcm = lcm*n//math.gcd(lcm, n)
        return lcm

    def get_true_N(self, N, stride_unrolls, portion_unrolls, unalignment_factor=1.0, page_size_bytes=None):
        # Align by vector size or page size
        if page_size_bytes is None:
            align_values = self.experiment.constants.simd_vec_values
        else:
            align_values = page_size_bytes // self.experiment.constants.dtype_size_bytes

        
        if type(stride_unrolls) == list and type(portion_unrolls) == list and len(stride_unrolls) == len(portion_unrolls):
            dimensions = stride_unrolls + portion_unrolls
        elif type(stride_unrolls) != list and type(portion_unrolls) != list:
            dimensions = [stride_unrolls, portion_unrolls]
        else:
            Logger.warn("Number of stride unroll configurations must match the number of portion unroll configurations.")
            return None

        lcm = self.lcm(dimensions)

        N = int(N * unalignment_factor)
        N = N // align_values
        N = N // lcm
        N = N * lcm
        N = N * align_values

        if N == 0:
            Logger.warn("Could not find dimensions for N.")
            return None

        return N

    def generate(self, configurations, compiler):
        commands = []
        test_functions = []
        true_configurations = []

        with multiprocessing.Pool() as pool:
            compilers = [compiler.copy(dmacro=configuration[0]) for configuration in configurations]
            gen_config = [configuration[1] for configuration in configurations]
            
            for context in pool.map(self.build_wrapper, zip(compilers, gen_config)):
                if not context is None:
                    if not context[0] is None:
                        commands += context[0]
                    if not context[1] is None:
                        test_functions += context[1]
                    if not context[2] is None:
                        true_configurations.append(context[2])

        return (commands, test_functions, true_configurations)
    
    def build_wrapper(self, configuration):
        self.compiler = configuration[0]
        true_configuration = self.build(configuration[1])
        return (self.commands, self.test_functions, true_configuration)

    def build(self, configurations, testing=False):
        Logger.warn(f"\"build\" not implemented for kernel {self.kernel_name}.")

    def aligned(self, bytes):
        try:
            assert(bytes % self.experiment.constants.simd_vec_bytes == 0)
        except Exception as e:
            print(f"{bytes} % {self.experiment.constants.simd_vec_bytes} != 0")
            raise e
        return bytes
    
    def z(self, bytes):
        return '' if bytes == 0 else bytes
    
    def aligned_z(self, bytes):
        return self.z(self.aligned(bytes))
    
    def register_command(self, command):
        self.commands.append(command)
    
    def register_test_function(self, test_function):
        self.test_functions.append(test_function)

    def get_test_functions(self):
        return self.test_functions
    
    def store_arrays(self, path, *arrays):
        if os.path.exists(path):
            os.unlink(path)

        if arrays:
            data = arrays[0].flatten()
            for array in arrays[1:]:
                data = np.append(data, array.flatten())

        data.tofile(path)

    def get_test_input_filename(self, test_data_dir):
        return os.path.join(test_data_dir, self.experiment.constants.test_input_filename)

    def get_test_output_filename(self, test_data_dir):
        return os.path.join(test_data_dir, self.experiment.constants.test_output_filename)

    def write_test_input(self, test_data_dir, *arrays):
        self.store_arrays(self.get_test_input_filename(test_data_dir), *arrays)
        
    def write_test_output(self, test_data_dir, *arrays):
        self.store_arrays(self.get_test_output_filename(test_data_dir), *arrays)

class RegisterAllocator:
    def __init__(self, register_set, default_column):
        self.pos = 0
        self.register_map = {register[0][default_column]:register[0] for register in register_set}
        self.registers = [register[0][default_column] for register in register_set] # Preserve order
        self.callee_saved = [register[1] for register in register_set]
        self.availability = {}
        self.used_callee_saved = []
        
        for register in self.registers:
            self.availability[register] = True

    def reserve(self, register, size_column=None):
        if not self.availability[register]:
            Logger.warn(f"Reserving an unavailable register: {register}.")
            
        self.availability[register] = False

        pos = self.registers.index(register)
        if self.callee_saved[pos] and register not in self.used_callee_saved:
            self.used_callee_saved.append(register)

        if size_column is None:
            return (register, register)
        return (register, self.register_map[register][size_column])

    def release(self, register):
        if self.availability[register]:
            Logger.warn(f"Releasing an unavailable register: {register}.")
            
        self.availability[register] = True
        
    def obtain(self, size_column=None):
        callee_saved_positions = []

        pos = self.pos

        # First check all non-callee saved positions
        while pos != (self.pos + len(self.registers) - 1) % len(self.registers):
            if self.callee_saved[pos]:
                callee_saved_positions.append(pos)
            elif self.availability[self.registers[pos]]:
                break
            pos = (pos + 1) % len(self.registers)

        if pos == (self.pos + len(self.registers) - 1) % len(self.registers) and not self.availability[self.registers[pos]]:
            for pos in callee_saved_positions:
                if self.availability[self.registers[pos]]:
                    break
            if not self.availability[self.registers[pos]]:
                Logger.warn(f"No available register could be found .")
                return None
            
            # Do not change starting position
            # - This makes it more likely to re-use callee-saved register in the future (reducing stack use)
            # - This does not offset the register in such a way that makes it more likely to re-use recently freed non-callee-saved registers
                
        else:
            # Start at next position next time
            self.pos = (pos + 1) % len(self.registers)

        register = self.registers[pos]
        
        if self.callee_saved[pos] and register not in self.used_callee_saved:
            self.used_callee_saved.append(register)

        self.availability[register] = False

        if size_column is None:
            return (register, register)
        
        return (register, self.register_map[register][size_column])

class CodeContext:
    def __init__(self, generator, stride_unrolls, portion_unrolls, N_allocated, trueN, suffix="", test_function_configuration=None):
        self.generator = generator
        self.stride_unrolls = stride_unrolls
        self.portion_unrolls = portion_unrolls
        self.N_allocated = N_allocated
        self.trueN = trueN
        self.test_function_configuration = test_function_configuration

        self.label = 0
        self.code = []
        self.prepare_name(generator.kernel_name, stride_unrolls, portion_unrolls, N_allocated, trueN, suffix=suffix)

        self.default_register_allocator = RegisterAllocator(generator.experiment.constants.default_register_set, generator.experiment.constants.default_register_set_default_column)
        self.simd_register_allocator = RegisterAllocator(generator.experiment.constants.simd_register_set, generator.experiment.constants.simd_register_set_default_column)
        self.register_allocators = {
            "default": self.default_register_allocator,
            "simd": self.simd_register_allocator
        }

        self.variables = {}


    def __enter__(self):
        return self

    def prepare_name(self, kernel_name, stride_unrolls, portion_unrolls, N, trueN, suffix=""):
        attr = [kernel_name + (('-' + suffix) if suffix != "" else ""), stride_unrolls * portion_unrolls, stride_unrolls, N, trueN]
        self.output_name = '_'.join([str(x) for x in attr])

    @staticmethod
    def decode_name(name):
        configuration = {}
        
        format = r'([0-9a-zA-Z\-]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)'
        search = re.search(format, name)
          
        if not search is None:
            groups = search.groups()
            configuration["total_unrolls"] = int(groups[1])
            configuration["stride_unrolls"] = int(groups[2])
            configuration["portion_unrolls"] = configuration["total_unrolls"] // configuration["stride_unrolls"] if configuration["stride_unrolls"] > 0 else 0
            configuration["N"] = int(groups[3])
            configuration["trueN"] = int(groups[4])
        else:
            format = r'([0-9a-zA-Z\-]+)_([0-9]+)'
            search = re.search(format, name)

        if not search is None:
            groups = search.groups()
            configuration["code"] = groups[0]
            if not "trueN" in configuration:
                configuration["trueN"] = int(groups[1])
        else:
            return None
        
        return configuration


    
    def get_name(self, extension=''):
        if extension:
            return f"{self.output_name}.{extension}" 
        else:
            return self.output_name

    def get_final_code(self):
        func = self.generator.experiment.constants.entry_function

        code = []
        code.append(f"    .file \"{self.get_name('gen')}\"")
        code.append(f"    .text")
        code.append(f"    .align    16,0x90")
        code.append(f"    .globl {func}")
        code.append(f"{func}:")
        code.append(f"..B1.{self.get_label()}:")
        code.append(f"    .cfi_startproc")

        pushed_registers = []
        for register_allocator in self.register_allocators.values():
            for register in register_allocator.used_callee_saved:
                code.append(self.indent(f"pushq     %{register}"))
                pushed_registers.append(register)

        code += self.code

        for pushed_register in pushed_registers[::-1]:
            code.append(self.indent(f"popq     %{pushed_register}"))

        code.append(self.indent(f"mfence"))
        code.append(self.indent(f"ret"))

        code.append(f"    .cfi_endproc")
        code.append(f"    .type	{func},@function")
        code.append(f"    .size	{func},.-{func}")
        code.append(f"    .data")
        code.append(f"    .align 8")
        code.append(f"    .global N")
        code.append(f"N:")
        code.append(f"    .long	{hex(self.N_allocated & ((1 << self.generator.experiment.constants.simd_vec_bytes) - 1))},{hex(self.N_allocated >> self.generator.experiment.constants.simd_vec_bytes)}")
        code.append(f"    .type	N,@object")
        code.append(f"    .size	N,8")
        code.append(f"    .section .note.GNU-stack, \"\"")
        
        return '\n'.join(code)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if not exception_type is None:
            print("exception_type:", exception_type)
            print("exception_value:", exception_value)
            traceback.print_tb(exception_traceback)
        else:
            for variable, register in self.variables.items():
                Logger.warn(f"Variable {variable} is still mapped to register {register} in kernel {self.generator.kernel_name}.")

        code = self.get_final_code()

        base_dir = self.generator.experiment.constants.construct_resource_path(experiment_name=self.generator.experiment.experiment_name, kernel_name=self.generator.kernel_name)
        
        if not self.test_function_configuration is None:
            base_dir = os.path.join(base_dir, "test", self.get_name())
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

        asm_dir = os.path.join(base_dir, "asm")
        asm_path = os.path.join(asm_dir, self.get_name("s"))
        os.makedirs(asm_dir, exist_ok=True)
        with open(asm_path, 'w+') as asm_file:
            asm_file.write(code)

        bin_dir = os.path.join(base_dir, "bin")
        bin_path = os.path.join(bin_dir, self.get_name())
        os.makedirs(bin_dir, exist_ok=True)
        self.generator.compiler.compile(asm_path, bin_path)

        if not self.test_function_configuration is None:
            input_file = self.generator.get_test_input_filename(data_dir)
            output_file = self.generator.get_test_output_filename(data_dir)
            self.generator.register_test_function(functools.partial(self.test_function_configuration[0], self.test_function_configuration[1], data_dir))
            self.generator.register_command(f"{bin_path} {input_file} {output_file}")
        else:
            self.generator.register_command(bin_path)

        res_dir = os.path.join(base_dir, "res")
        os.makedirs(res_dir, exist_ok=True)

    def indent(self, stmt):
        return f"        {stmt}"

    # === DEFAULT REGISTERS ===
    def set_variable(self, register, variable, size_column=None):
        for register_allocator in self.register_allocators.values():
            if register in register_allocator.registers:
                self.variables[variable] = register_allocator.reserve(register, size_column)
                return self.variables[variable][1]
            
        Logger.warn(f"Register {register} does not exist.")
        return None
    
    def get_variable(self, variable, size_column=None):
        register = self.variables[variable][1]

        if size_column is None:
            return register
        
        for register_allocator in self.register_allocators.values():
            for registers in register_allocator.register_map.values():
                if register in registers:
                    return registers[size_column]        

    def unset_variable(self, variable):
        register = self.variables[variable][0]
        for register_allocator in self.register_allocators.values():
            if register in register_allocator.registers:
                register_allocator.release(register)
                del self.variables[variable]
                return

    def unset_register(self, register):
        variables = []
        for variable, register_ in self.variables.items():
            if register_[0] == register:
                variables.append(variable)

        for variable in variables:
            self.unset_variable(self, variable)

    def get_register(self, register_set="default", variable=None, size_column=None):
        if not variable is None and variable in self.variables.keys():
            Logger.warn(f"Variable {variable} already exists in register {self.variables[variable][0]}, did you mean to use 'get_variable' instead?")

        register = self.register_allocators[register_set].obtain(size_column)
        if not variable is None:
            self.variables[variable] = register
        
        return register[1]
    
    # === CODE CONSTRUCTION ===

    def get_label(self):
        self.label += 1
        return self.label
    
    def add_statement(self, operation, *operands, indent=True):
        statement = operation + (' ' * max(1, (10 - len(operation)))) + ', '.join(operands)
        
        if indent:
            statement = self.indent(statement)
        
        self.code.append(statement)
        
class For:
    def __init__(self, cc, limit):
        self.cc = cc
        self.reg = cc.get_register()
        self.start_label = cc.get_label()
        self.limit = limit
        
    def __enter__(self):
        self.cc.add_statement(f"xorq", f"%{self.reg}", f"%{self.reg}")
        self.cc.add_statement(f"..B1.{self.start_label}:", indent=False)
        self.cc.add_statement(f"incq", f"%{self.reg}")
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        end_label = self.cc.get_label()
        self.cc.add_statement(f"cmpq",f"{self.limit}",f"%{self.reg}")
        self.cc.add_statement(f"jb",f"..B1.{self.start_label}")
        self.cc.add_statement(f"..B1.{end_label}:", indent=False)