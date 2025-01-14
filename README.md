# Multi-striding Experimental Framework
The Multi-striding Experimental Framework is used to execute the experiments found in the associated paper "Multi-Strided Access Patterns to Boost Hardware Prefetching" authored by Blom et al..
This framework supports three experiments:
1. The first conducted experiment explores the usage of multi-striding in micro-kernels performing aligned, unaligned and non-temporal load and store data movement operations.
2. The second experiment consists of a comparison of different striding configurations for various relevant compute kernels, some taken from PolyBench [https://www.cs.colostate.edu/~pouchet/software/polybench/], while preserving the number of executed instructions across configurations.
The latter indicates that the same number of loads and stores are performed for all codes, even if these can be optimized out for some.
3. The last experiment compares striding configurations, where the aforementioned loads and stores are optimized out, to various state-of-the-art implementations.

This framework is used to execute these experiments with ease by automatically generating assembly codes through parameterized scripts, test the results of the performed operations and run baseline or control implementations.

# Prerequisites
Preset experiments in this framework depend on different libraries that must be installed beforehand.
- LLVM v20.0 with Polly enabled, providing CLang as well [https://polly.llvm.org/get_started.html].
- Halide v18.0.0 [https://github.com/halide/Halide].
- Intel Math Kernel Library v2024.2 [https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html].
- OpenBLAS v0.3.28 [https://github.com/OpenMathLib/OpenBLAS].
- OpenCV v4.10.0 [https://github.com/opencv/opencv].
- Python 3.8.10 with pandas, numpy and matplotlib.
- libpng16 v1.6.34
- C++ 17
- Experiments involving performance counters require `perf`.

Note that the paths to the clang executable; Halide, MKL, OpenBLAS and OpenCV directories and libpng include directory must be supplied in the configuration file config.py in the root directory (see Section "Configuration" below).

A modified version of the STREAM v5.10 benchmark [https://www.cs.virginia.edu/stream/] is provided in this framework to support hugepages, used in the data movement experiment.
Additionally, the data movement experiment will initiate the installation of the Intel Memory Latency Checker binary version 3.11 (MLC) benchmark [https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html] in the `src` directory.

To run the data movement experiment, at least 5 GB in Huge Pages must be available, check `/proc/meminfo`.
To set the number of huge pages, 10 for example, one can use `sudo echo 10 > /proc/sys/vm/nr_hugepages`.

# Commands
The framework is initiated by running `python3 main.py <mode> [options]` in the terminal.
There are modes for experimentation (`E`), running the test environment (`T`), solely plotting using existing results (`P`), uploading the full framework to machines (`U`), downloading the full framework including results from machines (`D`) and cleaning up the project (`C`).

## Execute experiments and testing
To start the experiments, the user provides the `E` mode.
The user chooses a machine from the configuration file (see Section "Configuration" below) and one or multiple experiments to execute.
Some machines do not support pandas and matplotlib used in plotting.
Therefore the separate mode `P` can be used to plot on a different machine after downloading the results.
For example, `python3 main.py E mblom data_movement,compute` is used to run the `data_movement` and `compute` experiments using the `mblom` machine configuration.

To run tests associated to some experiments, using a machine configuration, the user simply uses the `T` mode instead of the `E` mode.
For example, `python3 main.py T mblom data_movement,compute`.

Using the argument `-g` will compile, when applicable, all binaries in debugging mode for both the experimentation and testing modes.
For example, `python3 main.py E mblom data_movement,compute -g`, will compile all associated sources with debugging flags and execute the generated binaries.

## Plotting
To plot results without building and executing binaries, the `P` mode is used.
This comes in use when pandas or matplotlib are not supported on a certain machine.
For example, `python3 main.py P mblom data_movement,compute` will plot all results from the `data_movement` and `compute` experiments for the `mblom` machine configuration.
The results for these experiments are assumed to be present locally.

## Uploading and downloading
To upload or download the entire directory to a specific machine through `scp`, the `U` and `D` options are used respectively.
For example, `python3 main.py U mblom` and `python3 main.py D mblom` will upload to and download from the remote location as specified in the configuration file (see Section "Configuration" below).

## Cleaning
As mentioned before, there is an option to clean up the project using the `C` mode.
If `-a` is supplied as an additional argument, also the `resources` and `results` directories are removed.
For example, this is achieved by running `python3 main.py C -a`.

# Structure and Reusability of the framework
The structure of this project allows for extension with additional experiments to investigate the effect of striding configurations.
In the root directory are a configuration file (`config.py`), the main script used to start up the framework and several directories.
The directories include:
- `classes`: containing base classes used in the framework.
- `compilers`: where users can supply specific compilation logic by inherriting from the compiler base class. A compiler which performs additional steps required by Halide is already provided.
- `execution_managers`: where users can supply classes that manage the execution of binaries. A default manager and a SLURM manager are already supplied.
- `experiments`: where users can define experiments. The configuration, method of execution, method of plotting results, testing individual kernel configurations, and executing control implementations or baselines is handled here. The experiments used in our paper are given.
- `generators`: where users can define how the x86-64 assembly code for individual kernels is to be generated given a striding configuration and the dimensions of datastructures. The generation of test data is also handled in these files. These resources are stored in the `resources` directory for the experiment initiating this generation. All kernels with functionality for testing is given.
- `resources`: this directory is made upon usage and contains assemblies, binaries, results and test data used in experimentation, plotting and testing. Resources are distributed over directories depending on the specified machine configuration and experiments.
- `results`: this directory is made upon usage and contains figures, csv and text files containing results that appear in our paper. These are the final results subdivided over directories indicating the used machines and executed experiments. 
- `src`: auxiliary files used in experimentation, such as source code, is stored here.

## Configuration
Both kernel configurations and machine configurations can be specified in the configuration file (as well as by deriving from the present base classes).
In the configuration file the user can supply constants, such as the size of the datatype and vectors used, the number of warm-up runs, repetitions and entries to be generated over which the median is computed in the experiments.
Functions for building paths within the `resource` and `results` directories are defined here.
Register sets for a specific architecture are defined, an x86_64 set is provided, also for AVX registers (AVX512, AVX2 and AVX).
After defining a new experiment, it should be registered in the experiment configurations dictionary.
Machine configurations are specified in a dictionary per using a name and a tuple of three dictionaries, containing specifications, in the following order:
- Machine specific configurations, such as the execution manager to be used, support for MSR, whether sudo can and must be used, and the remote address of this framework on that machine for uploading and downloading the framework including results.
- The aforementioned paths to prerequisite installations.
- Machine specific experiment configurations, these have formerly been acquired via experimentation and have been manually specified for the machines used in our experimentation.
The execution manager supplied to this machine configuration, for example the preset `das6` machines will use SLURM, so the corresponding execution manager class is supplied.

The configure function is called upon startup and will configure the machines, compiler and runtime arguments, constants, and experiments used in the project.


