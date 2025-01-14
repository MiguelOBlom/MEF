from .base.datacopy import DataCopyGenerator

class StreamReadAlignedWriteCopyGenerator(DataCopyGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "streamreadalignedwritecopy", testing=testing)

    def build_main_operation_load(self, vec, mem):
        return (f"vmovntdqa", mem, vec)
    
    def build_main_operation_store(self, vec, mem):
        return (f"vmovaps", vec, mem)