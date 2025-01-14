import os
from classes import Experiment, Logger
from generators import BicGGenerator, Convolution3x3Generator, DoitgenGenerator, GemverMxVGenerator, GemverMxVTGenerator, GemverSumGenerator, GemverOuterGenerator, Jacobi2DGenerator, MxVGenerator

can_plot = True
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except:
    can_plot = False

class ComputeExperiment(Experiment):
    def __init__(self, constants, compilers, machine_specific_experiment_configuration):
        super().__init__("compute", constants, compilers, machine_specific_experiment_configuration)

        
    def configure(self, testing=False):
        if testing:
            side = 256
            striding_configurations = [(3, 5)]

            unalignment_factor = 16/17
            self.bicg_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side}) for i, j in striding_configurations]
            self.convolution3x3_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side + 2}) for i, j in striding_configurations]
            self.doitgen_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side, "R": side, "Q": 1}) for i, j in striding_configurations]
            self.gemvermxv_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemvermxvt_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemverouter_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemversum_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.jacobi2d_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side + 2, "S": 1}) for i, j in striding_configurations]
            self.mxv_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            

        else:
            side = 32768
            striding_configurations = [(d, n // d) for n in range(1, 50) for d in self.get_divisors(n)]

            unalignment_factor = 1
            self.bicg_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side}) for i, j in striding_configurations]
            self.convolution3x3_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side//2 + 2}) for i, j in striding_configurations]
            self.doitgen_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side, "R": side, "Q": 1}) for i, j in striding_configurations]
            self.gemvermxv_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemvermxvt_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemverouter_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]
            self.gemversum_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": 536870912}) for i, j in striding_configurations]
            self.jacobi2d_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "X": side//2 + 2, "S": 1}) for i, j in striding_configurations]
            self.mxv_configurations = [({}, {"stride_unrolls": i, "portion_unrolls": j, "unalignment_factor": unalignment_factor, "P": side}) for i, j in striding_configurations]


        self.generator_configurations = [
                                         (BicGGenerator, self.bicg_configurations),
                                         (Convolution3x3Generator, self.convolution3x3_configurations),
                                         (DoitgenGenerator, self.doitgen_configurations),
                                         (GemverMxVGenerator, self.gemvermxv_configurations),
                                         (GemverMxVTGenerator, self.gemvermxvt_configurations),
                                         (GemverOuterGenerator, self.gemverouter_configurations),
                                         (GemverSumGenerator, self.gemversum_configurations),
                                         (Jacobi2DGenerator, self.jacobi2d_configurations),
                                         (MxVGenerator, self.mxv_configurations)
        ]

    def run(self):
        # Figure 6
        self.configure()
        compiler = self.compilers["minimal"].copy()
        compiler.infiles.append(os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c'))

        for generator_class, configurations in self.generator_configurations:
            for configuration in configurations:
                configuration[0]["N"] = generator_class.get_size_to_allocate(configuration[1])
            
            generator = generator_class(self)
            commands, _, _ = generator.generate(configurations, compiler)
            
            perf_commands = []
            event = "duration_time"
            args = ""
            for command in commands:
                sudo = ""
                if self.constants.machine_config.use_sudo:
                    sudo = "sudo"
                perf_commands.append(f"{sudo} {args} taskset -c 0 perf stat -x , -e {event}:u {command}")

            self.constants.machine_config.execution_manager.run(zip(commands, perf_commands), self.constants.entries, swap_stdout=True)


    def plot(self):
        if can_plot:
            result_dir = self.constants.construct_result_path(self.experiment_name)
            os.makedirs(result_dir, exist_ok=True)

            self.configure()
            configurations = self.get_result_configurations()
            for configuration in configurations:
                df = pd.read_csv(configuration["path"], names=["value", "a", "event", "b", "c", "d", "e", "f"])
                configuration["throughput"] = ((configuration["trueN"] * self.constants.dtype_size_bytes) / 1024**3) / (df["value"].median() / 1000**3)

            df = pd.DataFrame(configurations)
            grouped = df.groupby('kernel_name')
            for kernel, values in grouped:
                df = values.sort_values(by="stride_unrolls")
                ax = values.sort_values(by="stride_unrolls").plot.scatter(x="stride_unrolls", y="throughput", title=kernel, c=values["total_unrolls"], cmap='viridis')
                cb = ax.collections[0].colorbar
                cb.set_label("total_unrolls")
                plt.savefig(os.path.join(result_dir, f"{kernel}.png"))
                df.to_csv(os.path.join(result_dir, f"{kernel}.csv"))
        

    def test(self):
        self.configure(testing=True)
        compiler = self.compilers["minimal"].copy()
        compiler.infiles.append(os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c'))
        compiler.dmacro.update({"TESTING": "", "WARMUP": 0, "REPETITIONS": 1})

        for generator_class, configurations in self.generator_configurations:
            for configuration in configurations:
                configuration[0]["N"] = generator_class.get_size_to_allocate(configuration[1])
            generator = generator_class(self, testing=True)
            commands, test_functions, _ = generator.generate(configurations, compiler)
            self.constants.machine_config.execution_manager.run(commands, 1, test_functions=test_functions)
    