from .base.datamovement import DataMovementGenerator

class AlignedReadGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "alignedread", testing=testing)

    def build_main_operation(self, vec, mem):
        return (f"vmovaps", mem, vec)