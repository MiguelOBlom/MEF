from .base.datamovement import DataMovementGenerator

class StreamReadGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "streamread", testing=testing)

    def build_main_operation(self, vec, mem):
        return (f"vmovntdqa", mem, vec)