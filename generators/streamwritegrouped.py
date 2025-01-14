from .base.datamovement import DataMovementGenerator

class StreamWriteGroupedGenerator(DataMovementGenerator):
    def __init__(self, experiment, testing=False):
        super().__init__(experiment, "streamwritegrouped", testing=testing)

    def build_main_operation(self, vec, mem):
        return (f"vmovntdq", vec, mem)