import re
import os
import subprocess
from classes import ExecutionManager
from datetime import datetime
from classes import Logger

class DirectExecutionManager(ExecutionManager):
    def __init__(self):
        super().__init__("direct")

    def run(self, commands, n_entries, test_functions=[], swap_stdout=False, suffix='', env=None):
        commands = list(commands)
        if test_functions != []:
            if len(test_functions) != len(commands):
                Logger.fail("Cannot test commands, list of test functions must be of same size as list of commands.")
        else:
            test_functions = [None] * len(commands)

        for command, test_function in zip(commands, test_functions):
            if not test_function is None:
                test_data_dir = test_function()
                if test_data_dir is None:
                    Logger.fail("Need to return test data directory in test function.")

            exception = None
            try:
                if type(command) == tuple:
                    embedded_command = command[1]
                    command = command[0]
                else:
                    embedded_command = command

                # Get paths to outputs
                executable_path = command.split(' ')[0]
                executable_name = os.path.basename(executable_path)
                suffix_used = f'-{suffix}' if suffix else ''
                result_name = '_'.join([f"{executable_name.split('_')[0]}{suffix_used}"] + executable_name.split('_')[1:])
                resdir = os.path.join(os.path.dirname(os.path.dirname(executable_path)), 'res')
                resfile_path = os.path.join(resdir, f"{result_name}.csv")
                errfile_path = os.path.join(resdir, f"{result_name}.err")

                if swap_stdout:
                    temp = resfile_path
                    resfile_path = errfile_path
                    errfile_path = temp

                # Run command and write output to files
                if not os.path.exists(resfile_path):
                    with open(errfile_path, 'a+') as err_file:
                        with open(resfile_path, 'w+') as out_file:
                            if swap_stdout:
                                out_file.write(f"============ {datetime.now()} ============\n")
                                out_file.flush()
                                os.fsync(out_file.fileno())
                            else:
                                err_file.write(f"============ {datetime.now()} ============\n")
                                err_file.flush()
                                os.fsync(err_file.fileno())
                            embedded_command = re.sub(' +', ' ', embedded_command).strip()
                            print(embedded_command)

                            for _ in range(n_entries):
                                return_object = subprocess.run(embedded_command.split(' '), stdout=out_file, stderr=err_file, env=env, shell=('=' in embedded_command))
                                retval = return_object.returncode
                                if not test_function is None:
                                    if retval != 0:
                                        Logger.warn(f"Test for {embedded_command} failed!")
                                        print(retval)
                                    else:
                                        Logger.ok(f"Test for {embedded_command} passed!")

                else:
                    Logger.ok(f"Skipping {command}: result already exists!")
            except Exception as e:
                print(e)
                exception = e 
            finally:
                if not test_function is None:
                    files = os.listdir(test_data_dir)
                    for file in files:
                        os.unlink(os.path.join(test_data_dir, file))

                if not exception is None:
                    raise exception


            