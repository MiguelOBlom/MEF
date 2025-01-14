from classes import Logger
from .base.datamovement import DataMovementGenerator

class AlignedWriteGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "alignedwrite", testing=testing, initzero=testing)

    def build_main_operation(self, vec, mem):
        return (f"vmovaps", vec, mem)