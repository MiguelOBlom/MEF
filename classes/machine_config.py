import subprocess
from .logger import Logger


class MachineConfig:
    def __init__(self, machine_name, execution_manager, remote=None, msr=False, use_sudo=False, runtime_arguments={}):
        self.machine_name = machine_name
        self.execution_manager = execution_manager
        self.remote = remote
        self.msr = msr
        self.use_sudo = use_sudo
        self.runtime_arguments = runtime_arguments

    def handle_msr(self, nohwpf):
        if self.msr:
            if nohwpf:
                run_command = f"sudo wrmsr -a 0x1a4 0xf"
            else:
                run_command = f"sudo wrmsr -a 0x1a4 0x0"

            subprocess.call(run_command.split(' '))

            if nohwpf:
                Logger.ok("Hardware prefetching is turned off!")
            else:
                Logger.ok("Hardware prefetching is turned on!")