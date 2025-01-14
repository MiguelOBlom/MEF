from classes import Generator, CodeContext, For, Logger
import numpy as np
import os
from .computebase import ComputeBaseGenerator

class Jacobi2DBaseGenerator(ComputeBaseGenerator):
    def __init__(self, experiment, kernel_name, testing=False):
        super().__init__(experiment, kernel_name, testing=testing)

