import re
import shutil
import os
import subprocess
import time

from classes import ExecutionManager, Logger

class SLURMExecutionManager(ExecutionManager):
    def __init__(self, batch_size=5, nodes_available=24):
        super().__init__("slurm")
        self.slurm_dir = None
        self.batch_size = batch_size
        self.nodes_available = nodes_available

    def set_slurm_dir(self, slurm_dir):
        self.slurm_dir = slurm_dir
        os.makedirs(slurm_dir, exist_ok=True)

    def get_own_queued(self):
        queue = subprocess.check_output(['squeue']).decode("utf-8").split('\n')
        my_queue = []
        for q in queue:
            if os.getlogin() in q:
                my_queue.append(q)
        return my_queue

    def queue_command(self, command, slurm_dir):
        # Limit the number of batched workloads
        while len(self.get_own_queued()) > self.nodes_available:
            time.sleep(1)
            
        # If it is not full, we can just batch        
        cmd = f"echo '#!/bin/sh'$'\n'\"{command}\"| sbatch --output={os.path.join(slurm_dir, 'slurm-%j.out')} --open-mode=append --error={os.path.join(slurm_dir, 'slurm-%j.err')}"
        print(cmd)
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output = ps.communicate()[0]
        print(output)


    def run(self, commands, n_entries, test_functions=[], swap_stdout=False, suffix=''):
        commands = list(commands)
        if self.slurm_dir is None:
            Logger.fail("Set SLURM dir for SLURMExecutionManager.")

        if test_functions != []:
            if len(test_functions) != len(commands):
                Logger.fail("Cannot test commands, list of test functions must be of same size as list of commands.")
        else:
            test_functions = [None] * len(commands)
        
        run_commands = []

        for command, test_function in zip(commands, test_functions):
            if type(command) == tuple:
                embedded_command = command[1]
                command = command[0]
            else:
                embedded_command = command

            # Get paths to output files
            executable_path = command.split(' ')[0]
            executable_name = os.path.basename(executable_path)
            resdir = os.path.join(os.path.dirname(os.path.dirname(executable_path)), 'res')
            suffix = f'-{suffix}' if suffix else ''
            result_name = '_'.join([f"{executable_name.split('_')[0]}{suffix}"] + executable_name.split('_')[1:])
            resfile_path = os.path.join(resdir, f"{result_name}.csv")
            errfile_path = os.path.join(resdir, f"{result_name}.err")

            embedded_command = re.sub(' +', ' ', embedded_command).strip()
            print(embedded_command)

            if swap_stdout:
                temp = resfile_path
                resfile_path = errfile_path
                errfile_path = temp

            if not os.path.exists(resfile_path):
                run_commands.append((f"for i in \"$(seq 1 {n_entries})\"; do {embedded_command} >> {resfile_path} 2>> {errfile_path}; done", test_function))
            else:
                Logger.ok(f"Skipping {command}: result already exists!")

        # Queue command for SLURM
        all_test_data_dirs = []

        for run_command_batch in [run_commands[i:i+self.batch_size] for i in range(0, len(run_commands), self.batch_size)]:
            test_data_dirs = []

            for test_function in [run_command[1] for run_command in run_command_batch]:
                if not test_function is None:
                    test_data_dir = test_function()
                    if test_data_dir is None:
                        Logger.fail("Need to return test data directory in test function.")
                    else:
                        test_data_dirs.append(test_data_dir)
            
            command = '; '.join([run_command[0] for run_command in run_command_batch])

            all_test_data_dirs += test_data_dirs
            
            self.queue_command(command, self.slurm_dir)
        
        # Check test outputs
        if len(all_test_data_dirs):
            while len(self.get_own_queued()) > 0:
                time.sleep(1)

            for test_data_dir in test_data_dirs:
                shutil.rmtree(test_data_dir, ignore_errors=True)

            all_passed = True
            for test_data_dir in all_test_data_dirs:
                test_res_dir = os.path.abspath(os.path.join(test_data_dir, "..", "res"))
                for file in os.listdir(test_res_dir):
                    if file.split('.')[1] == 'csv':
                        test_res_path = os.path.join(test_res_dir, file)
                        with open(test_res_path, 'r') as test_res_file:
                            test_res = test_res_file.read()
                            if 'FAIL' in test_res:
                                Logger.warn(f"Test {test_res_path} failed!")
                                all_passed = False
                            else:
                                Logger.ok(f"Test {file} passed!")

            if all_passed:
                Logger.ok(f"All tests passed!")
            else:
                Logger.warn(f"Some tests have failed!")

