from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .computebase import ComputeBaseGenerator

class GemverMxVBaseGenerator(ComputeBaseGenerator):
    def __init__(self, experiment, kernel_name, testing=False):
        super().__init__(experiment, kernel_name, testing=testing)

    def test(self, configuration, test_data_dir):
        P = configuration["P"]

        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        unalignment_factor = configuration["unalignment_factor"]

        trueP = self.get_true_N(P, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)
        
        a = ((np.random.randint(100, size=(trueP, trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        x = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        y = ((np.random.randint(100, size=(trueP)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = (self.get_remainder(configuration, {"P": trueP})).astype(self.experiment.constants.nd_type)

        
        self.write_test_input(test_data_dir, a, x, y, remainder)

        for i in range(trueP):
            for j in range(trueP):
                x[i] = x[i] + -1.1 * a[j][i] * y[j]

        self.write_test_output(test_data_dir, a, x, y, remainder)

        return test_data_dir