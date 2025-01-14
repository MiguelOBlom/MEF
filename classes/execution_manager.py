from .logger import Logger

class ExecutionManager():
    def __init__(self, execution_manager_name):
        self.execution_manager_name = execution_manager_name

    def run(self, commands, n_entries, test_functions=None):
        Logger.warn(f"\"run\" not implemented for execution manager {self.execution_manager_name}.")