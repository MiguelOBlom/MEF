import shutil
import os
from classes import Experiment, Logger
from generators import AlignedReadGenerator, UnalignedReadGenerator, StreamReadGenerator, AlignedWriteGenerator, UnalignedWriteGenerator, StreamWriteGroupedGenerator, StreamWriteInterleavedGenerator, AlignedReadAlignedWriteCopyGenerator, AlignedReadStreamWriteCopyGenerator, StreamReadAlignedWriteCopyGenerator, StreamReadStreamWriteCopyGenerator
import subprocess

can_plot = True
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except:
    can_plot = False

MLC_DOWNLOAD_URL = "https://downloadmirror.intel.com/834254/mlc_v3.11b.tgz"

class DataMovementExperiment(Experiment):
    def __init__(self, constants, compilers, machine_specific_experiment_configuration):
        super().__init__("data_movement", constants, compilers, machine_specific_experiment_configuration)

    def run_throughput(self, generator_classes, configurations, minimal_compiler, machine_config, entries, do_nohwpf=True):
        for generator_class in generator_classes:
            generator = generator_class(self)
            compiler = minimal_compiler.copy(dmacro={"TIME": "", "MMAP_FLAG_HUGE": ""})
            commands, _, _ = generator.generate(configurations, compiler)
            if machine_config.msr:
                machine_config.handle_msr(False)

            machine_config.execution_manager.run(commands, entries, suffix="hwpf-throughput")
            
            if do_nohwpf and machine_config.msr:
                machine_config.handle_msr(True)
                machine_config.execution_manager.run(commands, entries, suffix="nohwpf-throughput")
                machine_config.handle_msr(False)

    def run_event(self, generator_classes, configurations, events, minimal_compiler, machine_config, entries, do_nohwpf=True):
        for generator_class in generator_classes:
            generator = generator_class(self)
            compiler = minimal_compiler.copy(dmacro={"MMAP_FLAG_HUGE": ""})
            commands, _, _ = generator.generate(configurations, compiler)
            for event in events:
                perf_commands = []
                args = ""
                for command in commands:
                    sudo = ""
                    if machine_config.use_sudo:
                        sudo = "sudo"
                    perf_commands.append(f"{sudo} {args} taskset -c 0 perf stat -x , -e {event}:u {command}")

                event_name = event.replace('.', '').replace('_', '').replace('-', '')

                if machine_config.msr:
                    machine_config.handle_msr(False)

                machine_config.execution_manager.run(zip(commands, perf_commands), entries, suffix=f"hwpf-{event_name}", swap_stdout=True)
                
                if do_nohwpf and machine_config.msr:
                    machine_config.handle_msr(True)
                    machine_config.execution_manager.run(zip(commands, perf_commands), entries, suffix=f"nohwpf-{event_name}", swap_stdout=True)
                    machine_config.handle_msr(False)


    def run(self):
        compiler = self.compilers["minimal"].copy(infiles=[os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c')], dmacro={"REPETITIONS": 10, "WARMUP": 2})

        # Figure 2
        read_write_copy_generators = [
            AlignedReadGenerator,
            UnalignedReadGenerator,
            StreamReadGenerator,
            AlignedWriteGenerator,
            UnalignedWriteGenerator,
            StreamWriteGroupedGenerator,
            StreamWriteInterleavedGenerator,
            AlignedReadAlignedWriteCopyGenerator,
            AlignedReadStreamWriteCopyGenerator,
            StreamReadAlignedWriteCopyGenerator,
            StreamReadStreamWriteCopyGenerator
        ]
        N = 536870912
        striding_configurations = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        unalignment_factor = 16/17
        configurations = [({"N": N}, {"suffix": "approx2GB", "N": N, "unalignment_factor": unalignment_factor, "stride_unrolls": sc[0], "portion_unrolls": sc[1]}) for sc in striding_configurations]
        self.run_throughput(read_write_copy_generators, configurations, compiler, self.constants.machine_config, self.constants.entries)
        self.run_stream()
        self.run_mlc()
        
        # Figure 3
        generators = [AlignedReadGenerator]
        N = 536870912
        striding_configurations = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        unalignment_factor = 16/17
        events = ["cycle_activity.stalls_l1d_miss",
                  "cycle_activity.stalls_l2_miss",
                  "cycle_activity.stalls_l3_miss",
                  "cycle_activity.stalls_mem_any",
                  "cycle_activity.stalls_total"]
        configurations = [({"N": N}, {"suffix": "approx2GB", "N": N, "unalignment_factor": unalignment_factor, "stride_unrolls": sc[0], "portion_unrolls": sc[1]}) for sc in striding_configurations]
        self.run_event(generators, configurations, events, compiler, self.constants.machine_config, self.constants.entries, do_nohwpf=False)
        
        # Figure 4
        generators = [AlignedReadGenerator]
        N = 536870912
        striding_configurations = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        unalignment_factor = 16/17
        events = ["L1-dcache-load-misses",
                  "L1-dcache-loads",
                  "LLC-load-misses",
                  "LLC-loads",
                  "LLC-store-misses",
                  "LLC-stores"]
        configurations = [({"N": N}, {"suffix": "approx2GB", "N": N, "unalignment_factor": unalignment_factor, "stride_unrolls": sc[0], "portion_unrolls": sc[1]}) for sc in striding_configurations]
        self.run_event(generators, configurations, events, compiler, self.constants.machine_config, self.constants.entries)

        # Figure 5
        generators = [
            AlignedReadGenerator,
            UnalignedReadGenerator,
            StreamReadGenerator,
            AlignedWriteGenerator,
            UnalignedWriteGenerator,
            StreamWriteGroupedGenerator,
        ]
        N = 536870912
        striding_configurations = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        unalignment_factor = 1
        configurations = [({"N": N}, {"suffix": "approx2GB", "N": N, "unalignment_factor": unalignment_factor, "stride_unrolls": sc[0], "portion_unrolls": sc[1]}) for sc in striding_configurations]
        self.run_throughput(read_write_copy_generators, configurations, compiler, self.constants.machine_config, self.constants.entries, do_nohwpf=False)
    
    def plot(self):
        if can_plot:
            result_dir = self.constants.construct_result_path(self.experiment_name)

            configurations = self.get_result_configurations()
            
            for configuration in configurations:
                if "throughput" in configuration["code"]:
                    df = pd.read_csv(configuration["path"], names=["throughput"])
                    configuration["event"] = "throughput"
                    configuration["value"] = df["throughput"].median()
                else:
                    df = pd.read_csv(configuration["path"], names=["value", "a", "event", "b", "c", "d", "e", "f"])
                    configuration["event"] = df["event"][0]
                    configuration["value"] = df["value"].median()

            df = pd.DataFrame(configurations)
            grouped = df.groupby('code')
            for kernel, values in grouped:
                values.sort_values(by="stride_unrolls").plot(x="stride_unrolls", y="value", title=kernel)
                plt.savefig(os.path.join(result_dir, f"{kernel}.png"))
                df.to_csv(os.path.join(result_dir, f"{kernel}.csv"))

    def test(self):
        N = 1024
        striding_configurations = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        unalignment_factor = 16/17
        configurations = [({"N": N}, {"suffix": "approx2GB", "N": N, "unalignment_factor": unalignment_factor, "stride_unrolls": sc[0], "portion_unrolls": sc[1]}) for sc in striding_configurations]
        compiler = self.compilers["minimal"].copy()
        compiler.infiles.append(os.path.join(self.constants.realpath, 'src', 'multistriding', 'main.c'))
        compiler.warmup = 0
        compiler.repetitions = 1
        compiler.dmacro.update({"TESTING": ""})
        self.run_tests([AlignedWriteGenerator, AlignedReadAlignedWriteCopyGenerator], compiler, configurations)

    def run_stream(self):
        src_dir = os.path.join(self.constants.realpath, "src")
        stream_dir = os.path.join(src_dir, "STREAM")
        stream_path = os.path.join(stream_dir, "stream_c.exe")
        
        if not os.path.exists(stream_path):
            command = f"make -C {stream_dir}"
            subprocess.run(command.split(' '))
        
        stream_res_dir = self.constants.construct_result_path(self.experiment_name)
        os.makedirs(stream_res_dir, exist_ok=True)
        stream_res_path = os.path.join(stream_res_dir, "STREAM.txt")
        with open(stream_res_path, 'w+') as stream_res_file:
            command = f"{stream_path} -DSTREAM_TYPE=float"
            subprocess.call(command.split(' '), stdout=stream_res_file)

    def run_mlc(self):
        src_dir = os.path.join(self.constants.realpath, "src")
        mlc_file = os.path.basename(MLC_DOWNLOAD_URL)
        mlc_version = '.'.join(mlc_file.split('.')[:-1])
        mlc_dir = os.path.join(src_dir, mlc_version)
        mlc_path = os.path.join(mlc_dir, "Linux", "mlc")
        
        if not os.path.exists(mlc_dir):
            os.makedirs(mlc_dir)

            tar_file = os.path.join(mlc_dir, mlc_file)
            
            commands = [
                f"wget {MLC_DOWNLOAD_URL} -P {mlc_dir}",
                f"tar -xzvf {tar_file} -C {mlc_dir}",
            ]

            for command in commands:
                subprocess.run(command.split(' '))

            os.unlink(tar_file)

        mlc_res_dir = self.constants.construct_result_path(self.experiment_name)
        os.makedirs(mlc_res_dir, exist_ok=True)
        mlc_res_path = os.path.join(mlc_res_dir, "MLC.txt")
        with open(mlc_res_path, 'w+') as mlc_res_file:
            command = f"{mlc_path} --max_bandwidth -k0 -Y -e"
            subprocess.call(command.split(' '), stdout=mlc_res_file)
    
    @staticmethod
    def clean(realpath):
        src_dir = os.path.join(realpath, "src")

        stream_dir = os.path.join(src_dir, "STREAM")
        stream_path = os.path.join(stream_dir, "stream_c.exe")

        if os.path.exists(stream_path):
            command = f"make clean -C {stream_dir}"
            subprocess.run(command.split(' '))

        mlc_file = os.path.basename(MLC_DOWNLOAD_URL)
        mlc_version = '.'.join(mlc_file.split('.')[:-1])
        mlc_dir = os.path.join(src_dir, mlc_version)
        if os.path.exists(mlc_dir):
            shutil.rmtree(mlc_dir, ignore_errors=True)

