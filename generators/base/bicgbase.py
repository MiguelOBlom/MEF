from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .computebase import ComputeBaseGenerator

class BicGBaseGenerator(ComputeBaseGenerator):
    def __init__(self, experiment, kernel_name, testing=False):
        super().__init__(experiment, kernel_name, testing=testing)

    def test(self, configuration, test_data_dir):
        X = configuration["X"]
        stride_unrolls = configuration["stride_unrolls"]
        portion_unrolls = configuration["portion_unrolls"]
        portion_unroll_values = portion_unrolls * self.experiment.constants.simd_vec_values
        unalignment_factor = configuration["unalignment_factor"]
        trueX = self.get_true_N(X, stride_unrolls, portion_unrolls, unalignment_factor=unalignment_factor)

        a = ((np.random.randint(100, size=(trueX,trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        s = ((np.random.randint(100, size=(trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        q = ((np.random.randint(100, size=(trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        p = ((np.random.randint(100, size=(trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        r = ((np.random.randint(100, size=(trueX)) / 10) - 5).astype(self.experiment.constants.nd_type)
        remainder = self.get_remainder(configuration, {"X": trueX}).astype(self.experiment.constants.nd_type)

        self.write_test_input(test_data_dir, a, s, q, p, r, remainder)

        for i in range(trueX):
            s[i] = 0

        for i in range(trueX):
            q[i] = 0                    

            for j in range(trueX):
                offset_s = j * self.experiment.constants.simd_vec_values
                s[j] = s[j] + r[i] * a[i][j]

                q[i] = q[i] + a[i][j] * p[j]
        
        self.write_test_output(test_data_dir, a, s, q, p, r, remainder)

        return test_data_dir
    
