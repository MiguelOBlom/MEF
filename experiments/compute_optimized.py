import copy
import re
import os
import functools
from classes import Experiment, Logger
from generators import BicGOptGenerator, Convolution3x3Generator, DoitgenOptGenerator, GemverOptGenerator, GemverMxVOptGenerator, GemverMxVTOptGenerator, GemverSumGenerator, GemverOuterOptGenerator, Jacobi2DOptGenerator, MxVOptGenerator
from compilers import HalideCompiler
from execution_managers import SLURMExecutionManager

can_plot = True
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except:
    can_plot = False

class ComputeOptimizedExperiment(Experiment):
    def __init__(self, constants, compilers, machine_specific_experiment_configuration):
        super().__init__("compute_optimized", constants, compilers, machine_specific_experiment_configuration)
    
    def configure(self, mode, testing=False):
        if testing:
            side = 512

            doitgen_P = 128
            doitgen_Q = 2
            doitgen_R = 128
            gemversum_P = side
            unalignment_factor = 16/17

            if mode == "nounroll":
                stride_unrolls_init = 1
                portion_unrolls_init = 1
                stride_unrolls_write = 1
                portion_unrolls_write = 1
                stride_unrolls_mxv = 1
                portion_unrolls_mxv = 1
                stride_unrolls_mxvt = 1
                portion_unrolls_mxvt = 1
                stride_unrolls_outer = 1
                portion_unrolls_outer = 1
                stride_unrolls_sum = 1
                portion_unrolls_sum = 1
                striding_configurations = [(1, 1)]
                gemveropt_mode = 0
            elif mode == "singlestrided":
                stride_unrolls_init = 1
                portion_unrolls_init = 5
                stride_unrolls_write = 1
                portion_unrolls_write = 5
                stride_unrolls_mxv = 1
                portion_unrolls_mxv = 5
                stride_unrolls_mxvt = 1
                portion_unrolls_mxvt = 5
                stride_unrolls_outer = 1
                portion_unrolls_outer = 5
                stride_unrolls_sum = 1
                portion_unrolls_sum = 5
                striding_configurations = [(1, 5)]
                gemveropt_mode = 1
            elif mode == "multistrided":
                stride_unrolls_init = 3
                portion_unrolls_init = 5
                stride_unrolls_write = 3
                portion_unrolls_write = 5
                stride_unrolls_mxv = 3
                portion_unrolls_mxv = 5
                stride_unrolls_mxvt = 3
                portion_unrolls_mxvt = 5
                stride_unrolls_outer = 3
                portion_unrolls_outer = 5
                stride_unrolls_sum = 3
                portion_unrolls_sum = 5
                striding_configurations = [(3, 5)]
                gemveropt_mode = 2
        else:
            side = 32768
            doitgen_P = side//2
            doitgen_Q = 4
            doitgen_R = 64
            gemversum_P = 536870912
            unalignment_factor = 1
            
            if mode == "nounroll":
                stride_unrolls_init = 1
                portion_unrolls_init = 1
                stride_unrolls_write = 1
                portion_unrolls_write = 1
                stride_unrolls_mxv = 1
                portion_unrolls_mxv = 1
                stride_unrolls_mxvt = 1
                portion_unrolls_mxvt = 1
                stride_unrolls_outer = 1
                portion_unrolls_outer = 1
                stride_unrolls_sum = 1
                portion_unrolls_sum = 1
                striding_configurations = [(1, 1)]
                gemveropt_mode = 0
            elif mode == "singlestrided":
                msec = self.machine_specific_experiment_configuration
                stride_unrolls_init = 1
                portion_unrolls_init = msec["single_portion_unrolls_init"]
                stride_unrolls_write = 1
                portion_unrolls_write = msec["single_portion_unrolls_write"]
                stride_unrolls_mxv = 1
                portion_unrolls_mxv = msec["single_portion_unrolls_mxv"]
                stride_unrolls_mxvt = 1
                portion_unrolls_mxvt = msec["single_portion_unrolls_mxvt"]
                stride_unrolls_outer = 1
                portion_unrolls_outer = msec["single_portion_unrolls_outer"]
                stride_unrolls_sum = 1
                portion_unrolls_sum = msec["single_portion_unrolls_sum"]
                striding_configurations = [(1, n) for n in range(1, 50)]
                gemveropt_mode = 1
            elif mode == "multistrided":
                msec = self.machine_specific_experiment_configuration
                stride_unrolls_init = msec["multi_stride_unrolls_init"] 
                portion_unrolls_init = msec["multi_portion_unrolls_init"]
                stride_unrolls_write = msec["multi_stride_unrolls_write"] 
                portion_unrolls_write = msec["multi_portion_unrolls_write"]
                stride_unrolls_mxv = msec["multi_stride_unrolls_mxv"]
                portion_unrolls_mxv = msec["multi_portion_unrolls_mxv"]
                stride_unrolls_mxvt = msec["multi_stride_unrolls_mxvt"]
                portion_unrolls_mxvt = msec["multi_portion_unrolls_mxvt"]
                stride_unrolls_outer = msec["multi_stride_unrolls_outer"]
                portion_unrolls_outer = msec["multi_portion_unrolls_outer"]
                stride_unrolls_sum = msec["multi_stride_unrolls_sum"]
                portion_unrolls_sum = msec["multi_portion_unrolls_sum"]
                striding_configurations = [(d, n // d) for n in range(1, 50) for d in self.get_divisors(n)]
                gemveropt_mode = 2
            
        self.bicgopt_configurations = [({}, {"suffix": mode,
                                             "stride_unrolls": i, 
                                             "portion_unrolls": j, 
                                             "stride_unrolls_init": stride_unrolls_init, 
                                             "portion_unrolls_init": portion_unrolls_init, 
                                             "unalignment_factor": unalignment_factor, 
                                             "X": side}) for i, j in striding_configurations]
        
        self.convolution3x3_configurations = [({}, {"suffix": mode,
                                                    "stride_unrolls": i, 
                                                    "portion_unrolls": j, 
                                                    "unalignment_factor": unalignment_factor, 
                                                    "X": side//2}) for i, j in striding_configurations]
        
        self.doitgenopt_configurations = [({}, {"suffix": mode,
                                                "stride_unrolls": i, 
                                                "portion_unrolls": j, 
                                                "stride_unrolls_init": stride_unrolls_init, 
                                                "portion_unrolls_init": portion_unrolls_init, 
                                                "stride_unrolls_write": 1, 
                                                "portion_unrolls_write": 1, 
                                                "unalignment_factor": unalignment_factor, 
                                                "P": doitgen_P, "Q": doitgen_Q, "R": doitgen_R}) for i, j in striding_configurations]
        
        self.gemvermxvopt_configurations = [({}, {"suffix": mode,
                                                  "stride_unrolls": i, 
                                                  "portion_unrolls": j, 
                                                  "unalignment_factor": unalignment_factor, 
                                                  "P": side}) for i, j in striding_configurations]
        
        self.gemvermxvtopt_configurations = [({}, {"suffix": mode,
                                                    "stride_unrolls": i, 
                                                   "portion_unrolls": j, 
                                                   "unalignment_factor": unalignment_factor, 
                                                   "P": side}) for i, j in striding_configurations]
        
        self.gemverouteropt_configurations = [({}, {"suffix": mode,
                                                    "stride_unrolls": i, 
                                                    "portion_unrolls": j, 
                                                    "unalignment_factor": unalignment_factor, 
                                                    "P": side}) for i, j in striding_configurations]
        
        self.gemveropt_configurations = [({}, {"suffix": mode,
                                               "mode": gemveropt_mode,
                                               "stride_unrolls_mxv": stride_unrolls_mxv, 
                                               "portion_unrolls_mxv": portion_unrolls_mxv, 
                                               "stride_unrolls_mxvt": stride_unrolls_mxvt, 
                                               "portion_unrolls_mxvt": portion_unrolls_mxvt, 
                                               "stride_unrolls_outer": stride_unrolls_outer, 
                                               "portion_unrolls_outer": portion_unrolls_outer, 
                                               "stride_unrolls_sum": stride_unrolls_sum, 
                                               "portion_unrolls_sum": portion_unrolls_sum, 
                                               "unalignment_factor": unalignment_factor, 
                                               "P": side})]
        
        self.gemversum_configurations = [({}, {"suffix": mode,
                                               "stride_unrolls": i, 
                                               "portion_unrolls": j, 
                                               "unalignment_factor": unalignment_factor, 
                                               "P": gemversum_P}) for i, j in striding_configurations]
        
        self.jacobi2dopt_configurations = [({}, {"suffix": mode,
                                                 "stride_unrolls": i, 
                                                 "portion_unrolls": j, 
                                                 "stride_unrolls_write": stride_unrolls_write, 
                                                 "portion_unrolls_write": portion_unrolls_write, 
                                                 "unalignment_factor": unalignment_factor, 
                                                 "X": side//2 + 2, "S": 5}) for i, j in striding_configurations]
        
        self.mxvopt_configurations = [({}, {"suffix": mode,
                                            "stride_unrolls": i, 
                                            "portion_unrolls": j, 
                                            "unalignment_factor": unalignment_factor, 
                                            "P": side}) for i, j in striding_configurations]

        self.generator_configurations = {
                                         "mxvopt": (MxVOptGenerator, self.mxvopt_configurations),
                                         "jacobi2dopt": (Jacobi2DOptGenerator, self.jacobi2dopt_configurations),
                                         "convolution3x3": (Convolution3x3Generator, self.convolution3x3_configurations),
                                         "gemveropt": (GemverOptGenerator, self.gemveropt_configurations),
                                         "bicgopt": (BicGOptGenerator, self.bicgopt_configurations),
                                         "gemvermxvopt": (GemverMxVOptGenerator, self.gemvermxvopt_configurations),
                                         "gemvermxvtopt": (GemverMxVTOptGenerator, self.gemvermxvtopt_configurations),
                                         "gemverouteropt": (GemverOuterOptGenerator, self.gemverouteropt_configurations),
                                         "gemversum": (GemverSumGenerator, self.gemversum_configurations),
                                         "doitgenopt": (DoitgenOptGenerator, self.doitgenopt_configurations),
        }


    def run(self):
        # Figure 7
        for mode in ["nounroll", "singlestrided", "multistrided"]:
            self.configure(mode)
            compiler = self.compilers["minimal"].copy()
            compiler.infiles.append(os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c'))

            for kernel_name, generator_context in self.generator_configurations.items():
                generator_class, configurations = generator_context

                for configuration in configurations:
                    configuration[0]["N"] = generator_class.get_size_to_allocate(configuration[1])
                
                generator = generator_class(self)
                commands, _, true_configurations = generator.generate(configurations, compiler)
                control_commands_arguments = self.control(kernel_name, true_configurations)

                perf_commands = []
                event = "duration_time"
                all_commands = list(zip(commands, [''] * len(commands))) + control_commands_arguments
                binaries = []
                for command, args in all_commands:
                    sudo = ""
                    if self.constants.machine_config.use_sudo:
                        sudo = "sudo"
                    binaries.append(command)
                    perf_commands.append(f"{sudo} {args} taskset -c 0 perf stat -x , -e {event}:u {command}")
                self.constants.machine_config.execution_manager.run(zip(binaries, perf_commands), self.constants.entries, swap_stdout=True)

    def plot(self):
        if can_plot:
            result_dir = self.constants.construct_result_path(self.experiment_name)
            os.makedirs(result_dir, exist_ok=True)

            configurations = self.get_result_configurations()
            for configuration in configurations:
                df = pd.read_csv(configuration["path"], names=["value", "a", "event", "b", "c", "d", "e", "f"])
                try:
                    configuration["throughput"] = ((configuration["trueN"] * self.constants.dtype_size_bytes) / 1024**3) / (df["value"].median() / 1000**3)
                except Exception as e:
                    print(e)
                    continue


            df = pd.DataFrame(configurations)
            kernel_grouped = df.groupby('kernel_name')
            for kernel_name, kernel_values in kernel_grouped:
                code_grouped = kernel_values.groupby('code')

                results = []
                for code_name, code_values in code_grouped:
                    results.append({"code": code_name, "max_throughput": code_values["throughput"].max()})
                
                df = pd.DataFrame(results)

                df["speedup"] = df.loc[df["code"] == kernel_name + "-multistrided"]["max_throughput"].max() / df["max_throughput"]
                ax = df.plot.bar(x="code", y="speedup", title=f"Speedup of multistrided {kernel_name} over state-of-the-art")
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f")
                ax.get_legend().remove()
                x1,x2,y1,y2 = plt.axis()
                plt.axis((x1, x2, y1, y2 * 1.1))
                plt.subplots_adjust(bottom=0.5)
                plt.savefig(os.path.join(result_dir, f"{kernel_name}.png"))
                df.to_csv(os.path.join(result_dir, f"{kernel_name}.csv"))

    def test(self):
        self.configure(mode="multistrided", testing=True)
        compiler = self.compilers["minimal"].copy()
        compiler.infiles.append(os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c'))
        compiler.dmacro.update({"TESTING": "", "WARMUP": 0, "REPETITIONS": 1})

        for kernel_name, generator_context in self.generator_configurations.items():
            generator_class, configurations = generator_context

            for configuration in configurations:
                configuration[0]["N"] = generator_class.get_size_to_allocate(configuration[1])
            generator = generator_class(self, testing=True)
            commands, test_functions, true_configurations = generator.generate(configurations, compiler)
            input = commands[0].split(' ')[1]
            output = commands[0].split(' ')[2]
            testdir = os.path.dirname(input)

            self.constants.machine_config.execution_manager.run(commands, 1, test_functions=test_functions)

            control_commands_arguments = self.control(kernel_name, true_configurations, testing=True)
           
            for command, args in control_commands_arguments:
                if "halide" in command:
                    if kernel_name == "convolution3x3":
                        assert(len(true_configurations) == 1)
                        used_test_functions=[functools.partial(generator.test_halide, true_configurations[0], testdir)]
                    else:
                        # Similar displacement to convolution3x3
                        continue
                else:
                    used_test_functions = test_functions
                
                os.makedirs(testdir, exist_ok=True)

                if not type(self.constants.machine_config.execution_manager) == SLURMExecutionManager:
                
                    command = (f"{command}", f"{command} {input} {output}")

                    args = re.sub(' +', ' ', args).strip()

                    env = os.environ.copy()
                    if args:
                        for arg in args.split(' '):
                            name, val = arg.split('=')
                            if name == "LD_LIBRARY_PATH" and "LD_LIBRARY_PATH" in env.keys():
                                env["LD_LIBRARY_PATH"] = f"{env['LD_LIBRARY_PATH']}:{val}"
                            else:
                                env[name] = val

                    self.constants.machine_config.execution_manager.run([command], 1, test_functions=used_test_functions, env=env)
                else:
                    command = (f"{command}", f"{args} {command} {input} {output}")
                    self.constants.machine_config.execution_manager.run([command], 1, test_functions=used_test_functions)




    def control(self, kernel_name, true_configurations, testing=False):
        # Run state-of-the-art implementations
        commands = []
        true_configurations_immutable = [(trueN, N, tuple(sorted(d.items()))) for trueN, N, d in true_configurations]
        true_configurations_unique = list(set(true_configurations_immutable))
        true_configurations = [(trueN, N, dict(d)) for trueN, N, d in true_configurations_unique]

        kernel_base_dir = self.constants.construct_resource_path(experiment_name=self.experiment_name, kernel_name=kernel_name)
        if testing:
            kernel_base_dir = os.path.join(kernel_base_dir, "test")
            os.makedirs(os.path.join(kernel_base_dir, "res"), exist_ok=True)
        control_dir = os.path.join(kernel_base_dir, "control")
        os.makedirs(control_dir, exist_ok=True)

        control_src_dir = os.path.join(self.constants.realpath, "src", "kernels", kernel_name)
        control_srcs = os.listdir(control_src_dir)
        for control_src in control_srcs:
            control_name = control_src.split('.')[0]                
            control_src_path = os.path.join(control_src_dir, control_src)
            compiler = self.compilers[control_name].copy()
            if testing:
                compiler.dmacro.update({"TESTING": "", "WARMUP": 0, "REPETITIONS": 1})

            if control_name in self.constants.machine_config.runtime_arguments.keys():
                arguments = self.constants.machine_config.runtime_arguments[control_name]
            else:
                arguments = ''

            for trueN, N, true_dmacro in true_configurations:
                if type(compiler) == HalideCompiler:
                    side = true_dmacro["prebuild_side"]
                    del true_dmacro["prebuild_side"]

                if testing:
                    true_dmacro["testN"] = N
                
                control_bin_path = os.path.join(control_dir, f"{control_name}_{trueN}")
                configured_compiler = compiler.copy(dmacro=true_dmacro)

                if type(compiler) == HalideCompiler:
                    generators = [kernel_name]
                    if kernel_name == "jacobi2dopt":
                        generators.append("writeback")

                    includes_infiles = configured_compiler.prebuild(generators, str(side), str(trueN))
                    for include, infiles in includes_infiles:
                        configured_compiler = configured_compiler.copy(include=include, infiles=infiles)
                        autoscheduler = os.path.basename(include[0])
                        autoscheduler_control_bin_path = os.path.join(os.path.dirname(control_bin_path), f"{autoscheduler}{os.path.basename(control_bin_path)}")
                        configured_compiler.compile(control_src_path, autoscheduler_control_bin_path)
                        commands.append((autoscheduler_control_bin_path, arguments))
                else:
                    configured_compiler.compile(control_src_path, control_bin_path)
                    commands.append((control_bin_path, arguments))
        return commands




        



            

