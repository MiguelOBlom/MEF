from .base.datamovement import DataMovementGenerator

class UnalignedReadGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "unalignedread", testing=testing, unaligned=True)

    def build_main_operation(self, vec, mem):
        return (f"vmovups", mem, vec)