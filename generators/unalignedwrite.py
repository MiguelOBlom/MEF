from .base.datamovement import DataMovementGenerator

class UnalignedWriteGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "unalignedwrite", testing=testing, unaligned=True)

    def build_main_operation(self, vec, mem):
        return (f"vmovups", vec, mem)