from .base.datamovement import DataMovementGenerator

class StreamWriteInterleavedGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "streamwriteinterleaved", testing=testing, grouped=False)

    def build_main_operation(self, vec, mem):
        return (f"vmovntdq", vec, mem)