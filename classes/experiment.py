import os
from .logger import Logger
from .generator import CodeContext

class Experiment:
    def __init__(self, experiment_name, constants, compilers, machine_specific_experiment_configuration):
        self.experiment_name = experiment_name
        self.constants = constants
        self.compilers = compilers
        self.machine_specific_experiment_configuration = machine_specific_experiment_configuration

    def run(self):
        Logger.warn(f"\"run\" not implemented for experiment {self.experiment_name}.")

    def get_result_configurations(self):
        kernels_dir = os.path.join(self.constants.construct_resource_path(experiment_name=self.experiment_name), "kernels")
        
        configurations = []

        kernels = os.listdir(kernels_dir)
        for kernel in kernels:
            res_dir = os.path.join(kernels_dir, kernel, "res")
            res_dir_files = os.listdir(res_dir)
            for res_dir_file in res_dir_files:
                name, extension = res_dir_file.split('.')
                if extension == 'csv':
                    configuration = CodeContext.decode_name(name)
                    configuration["kernel_name"] = kernel
                    configuration["path"] = os.path.join(res_dir, res_dir_file)
                    configurations.append(configuration)

        return configurations

    def run_tests(self, generator_classes, compiler, configurations):
        for generator_class in generator_classes:
            generator = generator_class(self, testing=True)
            commands, test_functions, _ = generator.generate(configurations, compiler)
            self.constants.machine_config.execution_manager.run(commands, 1, test_functions=test_functions)

    def get_divisors(self, N):
        for i in range(1, N + 1):
            if N % i == 0:
                yield i

    @staticmethod
    def clean(realpath):
        pass