from .base.datacopy import DataCopyGenerator

class AlignedReadAlignedWriteCopyGenerator(DataCopyGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "alignedreadalignedwritecopy", testing=testing)

    def build_main_operation_load(self, vec, mem):
        return (f"vmovaps", mem, vec)
    
    def build_main_operation_store(self, vec, mem):
        return (f"vmovaps", vec, mem)