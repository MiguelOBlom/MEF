import os
import functools
from classes import MachineConfig, Constants, Compiler, Logger
from experiments import DataMovementExperiment, ComputeExperiment, ComputeOptimizedExperiment
from execution_managers import DirectExecutionManager, SLURMExecutionManager
import shutil
import multiprocessing
import subprocess
from compilers import HalideCompiler

ENTRY_FUNCTION="experiment"
TEST_INPUT_FILENAME="input.txt"
TEST_OUTPUT_FILENAME="output.txt"
DTYPE_SIZE_BYTES=4 # float
SIMD_VEC_SIZE_BITS=256 # AVX2
ND_TYPE='float32'
WARMUP=0
REPETITIONS=5
ENTRIES=5

def resource_path_construction(realpath, machine_name, experiment_name=None, kernel_name=None):
    base_dir = os.path.join(realpath, "resources", machine_name)
    
    if not experiment_name is None:
        base_dir = os.path.join(base_dir, "experiments", experiment_name)
    else:
        if not kernel_name is None:
            Logger.warn("kernel_name is set, but experiment_name is not.")
        return base_dir

    if not kernel_name is None:
        base_dir = os.path.join(base_dir, "kernels", kernel_name)

    return base_dir

def result_path_construction(realpath, machine_name, experiment_name):
    return os.path.join(realpath, "results", machine_name, experiment_name)

# Registers
x86_64_default_registers = [
    (["rax", "eax", "ax", "ah", "al"], False),
    (["rbx", "ebx", "bx", "bh", "bl"], True),
    (["rcx", "ecx", "cx", "ch", "cl"], False),
    (["rdx", "edx", "dx", "dh", "dl"], False),
    (["rsi", "esi", "si", None, "sil"], False),
    (["rdi", "edi", "di", None, "dil"], False),
    (["rbp", "ebp", "bp", None, "bpl"], True),
    (["rsp", "esp", "sp", None, "spl"], False),
    (["r8", "r8d", "r8w", None, "r8l"], False),
    (["r9", "r9d", "r9w", None, "r9l"], False),
    (["r10", "r10d", "r10w", None, "r10l"], False),
    (["r11", "r11d", "r11w", None, "r11l"], False),
    (["r12", "r12d", "r12w", None, "r12l"], True),
    (["r13", "r13d", "r13w", None, "r13l"], True),
    (["r14", "r14d", "r14w", None, "r14l"], True),
    (["r15", "r15d", "r15w", None, "r15l"], True),
]
x86_64_default_registers_default_column = 0

avx_simd_registers = [([f"{size}mm{no}" for size in ['z', 'y', 'x']], False) for no in range(16)]
avx_simd_registers_default_column = 1 # AVX2

# Experiments
experiment_configurations = {
    "data_movement": DataMovementExperiment,
    "compute": ComputeExperiment,
    "compute_optimized": ComputeOptimizedExperiment,
}

# Machines
machine_configs = {
    "local": ({"execution_manager": DirectExecutionManager, "msr": True, "use_sudo": True},
              {"clang": "clang"},
              {}
    ),
    "mblom": ({"execution_manager": DirectExecutionManager, "remote": "mblom:/home/blommo", "msr": True, "use_sudo": True},
              {"clang": "/usr/local/bin/clang",
               "halide": "/home/blommo/Halide",
               "libpng16": "/usr/include/libpng16",
               "mkl": "/opt/intel/oneapi/mkl/latest", 
               "openblas": "/home/blommo/OpenBLAS", 
               "opencv": "/home/blommo/opencv-4.x"},
              { "compute_optimized":
                {"multi_stride_unrolls_init": 5,
                "multi_portion_unrolls_init": 6,
                "multi_stride_unrolls_write": 5,
                "multi_portion_unrolls_write": 8,

                "single_portion_unrolls_init": 48,
                "single_portion_unrolls_write": 28,
                
                "multi_stride_unrolls_mxv": 7,
                "multi_portion_unrolls_mxv": 6,
                "multi_stride_unrolls_mxvt": 10,
                "multi_portion_unrolls_mxvt": 4,
                "multi_stride_unrolls_outer": 5,
                "multi_portion_unrolls_outer": 4,
                "multi_stride_unrolls_sum": 5,
                "multi_portion_unrolls_sum": 8,

                "single_portion_unrolls_mxv": 7,
                "single_portion_unrolls_mxvt": 6,
                "single_portion_unrolls_outer": 10,
                "single_portion_unrolls_sum": 49}
              }
    ),
    "das6": ({"execution_manager": SLURMExecutionManager, "remote": "das6:/home/blommo"},
              {"clang": "/usr/local/bin/clang",
               "halide": "/home/blommo/Halide",
               "libpng16": "/usr/include/libpng16",
               "mkl": "/opt/intel/oneapi/mkl/latest", 
               "openblas": "/home/blommo/OpenBLAS", 
               "opencv": "/home/blommo/opencv-4.x"},
              { "compute_optimized":
                {"multi_stride_unrolls_init": 3,
                "multi_portion_unrolls_init": 1,
                "multi_stride_unrolls_write": 5,
                "multi_portion_unrolls_write": 4,

                "single_portion_unrolls_init": 1,
                "single_portion_unrolls_write": 33,
                
                "multi_stride_unrolls_mxv": 2,
                "multi_portion_unrolls_mxv": 24,
                "multi_stride_unrolls_mxvt": 2,
                "multi_portion_unrolls_mxvt": 20,
                "multi_stride_unrolls_outer": 5,
                "multi_portion_unrolls_outer": 2,
                "multi_stride_unrolls_sum": 5,
                "multi_portion_unrolls_sum": 9,

                "single_portion_unrolls_mxv": 7,
                "single_portion_unrolls_mxvt": 11,
                "single_portion_unrolls_outer": 7,
                "single_portion_unrolls_sum": 33}
              }
    ),
}

def configure(machine_name, experiment_names, realpath, debug=False):
    construct_resource_path = functools.partial(resource_path_construction, realpath, machine_name)
    construct_result_path = functools.partial(result_path_construction, realpath, machine_name)

    # Machines
    machine_config, paths, machine_specific_experiment_configurations = machine_configs[machine_name]
    machine_config["execution_manager"] = machine_config["execution_manager"]()
    if type(machine_config["execution_manager"]) == SLURMExecutionManager:
        slurm_dir = os.path.join(construct_resource_path(), "slurm")
        machine_config["execution_manager"].set_slurm_dir(slurm_dir)
    else:
        machine_config["execution_manager"]
    machine_config["machine_name"] = machine_name

    # Compilers and runtime arguments
    compilers = {}

    env = os.environ.copy()
    runtime_arguments_map = {}
    
    if "clang" in paths.keys():
        compilers["minimal"] = Compiler(paths["clang"], debug=debug, warn=['all', 'extra'], opt=3, fopt=['no-inline'], include=[os.path.join(realpath, "src", "multistriding")], dmacro={"WARMUP": WARMUP, "REPETITIONS": REPETITIONS})
        compilers["default"] = compilers["minimal"].copy(mopt=["arch=native", "avx2"], fopt=["vectorize"], infiles=[os.path.join(realpath, "src", "multistriding", "main.c")])
        compilers["polly"] = compilers["default"].copy(mopt=["llvm -polly", "llvm -polly-vectorizer=stripmine"])
        
        if "openblas" in paths.keys():
            runtime_arguments_map["openblas"] = {
                "OPENBLAS_NUM_THREADS": "1",
            }


            compilers["openblas"] = compilers["default"].copy(include=[paths["openblas"]], lib=[paths["openblas"]], libraries=["openblas", "pthread"])
        
        if "mkl" in paths.keys():
            runtime_arguments_map["mkl"] = {
                "LD_LIBRARY_PATH": f"{env.get('LD_LIBRARY_PATH', '')}:{paths['mkl']}/lib",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }

            mkl_libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core", "m"]
            compilers["mkl"] = compilers["default"].copy(include=[os.path.join(paths["mkl"], "include")], lib=[os.path.join(paths["mkl"], "lib")], dmacro={"MKL_ILP64": ""}, libraries=mkl_libraries)
        
        if "opencv" in paths.keys():
            runtime_arguments_map["opencv"] ={
                "LD_LIBRARY_PATH": f"{env.get('LD_LIBRARY_PATH', '')}:{paths['opencv']}/build/lib/:{paths['mkl']}/lib",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }

            opencv_modules = ["core", "calib3d", "features2d", "flann", "dnn", "highgui", "imgcodecs", "videoio", "imgproc", "ml", "objdetect", "photo", "stitching", "video"]
            opencv_include = [os.path.join(paths["opencv"], path) for path in ["build", "include"] + [os.path.join("modules", module, "include") for module in opencv_modules]]
            opencv_lib = [os.path.join(paths["opencv"], "build", "lib"), os.path.join(paths["mkl"], "lib")]
            opencv_libraries = ["opencv_core", "opencv_imgproc", "stdc++"] + mkl_libraries
            if machine_name == "mblom" or machine_name == "das6":
                opencv_cc= "g++"
            else:
                opencv_cc = paths["clang"]

            compilers["opencv"] = compilers["default"].copy(cc=opencv_cc, include=opencv_include, lib=opencv_lib, libraries=opencv_libraries)
            compilers["opencv"].fopt.remove("vectorize")
        
        if "halide" in paths.keys():
            runtime_arguments_map["halide"] = {
                "LD_LIBRARY_PATH": f"{env.get('LD_LIBRARY_PATH', '')}:{paths['halide']}/bin",
                "HL_NUM_THREADS": "1",
            }

            halide_resource_dir = os.path.join(construct_resource_path(), "halide")
            halide_src_dir = os.path.join(realpath, "src", "halide")
            shutil.rmtree(halide_resource_dir, ignore_errors=True)

            halide_cc = paths["clang"]
            halide_include = [os.path.join(paths["halide"], "include"), os.path.join(paths["halide"], "tools"), paths["libpng16"]]
            halide_lib = [os.path.join(paths["halide"], "bin")]
            halide_libraries = ["Halide", "pthread", "dl"]
            if machine_name == "mblom":
                halide_libraries.append("png16")
                halide_libraries.append("jpeg")
            
            if machine_name == "das6":
                halide_cc = "g++"
                halide_standard = "c++17"
            else:
                halide_libraries.append("stdc++")
                halide_standard = ''

            halide_autoschedulers = [("Mullapudi2016", "libautoschedule_mullapudi2016.so"),
                                     ("Adams2019", "libautoschedule_adams2019.so"),
                                     ("Li2018", "libautoschedule_li2018.so")]

            halide_resource_compiler = compilers["minimal"].copy(
                    cc = halide_cc,
                    standard=halide_standard,
                    infiles=[os.path.join(paths["halide"], "tools", "GenGen.cpp")], 
                    fopt=["no-rtti"],
                    include=[os.path.join(paths["halide"], "include")],
                    lib=[os.path.join(paths["halide"], "bin")],
                    libraries=halide_libraries
                    )

            compilers["halide"] = HalideCompiler(halide_cc, paths["halide"], halide_src_dir, halide_resource_dir, halide_resource_compiler, halide_autoschedulers, include=halide_include, lib=halide_lib, libraries=halide_libraries)
            compilers["halide"].update_from_compiler(compilers["default"])
            compilers["halide"].fopt.remove("vectorize")
            compilers["halide"].cc = halide_cc
            compilers["halide"].standard = halide_standard
    else:
        Logger.fail("Cannot generate codes without a compiler specified in config.py")

    runtime_arguments = {library: ' '.join([f"{variable}={value}" for variable, value in arguments.items()]) for library, arguments in runtime_arguments_map.items()}       

    machine_config["runtime_arguments"] = runtime_arguments
    machine = MachineConfig(**machine_config)

    # Constants
    constants_configuration = Constants(ENTRY_FUNCTION,
                                        DTYPE_SIZE_BYTES, 
                                        SIMD_VEC_SIZE_BITS,
                                        ND_TYPE,
                                        WARMUP,
                                        REPETITIONS,
                                        ENTRIES,
                                        x86_64_default_registers, x86_64_default_registers_default_column, 
                                        avx_simd_registers, avx_simd_registers_default_column,
                                        machine,
                                        realpath,
                                        construct_resource_path,
                                        construct_result_path,
                                        TEST_INPUT_FILENAME, TEST_OUTPUT_FILENAME)

    # Experiments
    experiments = []
    for experiment_name in experiment_names:
        if experiment_name not in experiment_configurations.keys():
            Logger.fail(f"Experiment name \"{experiment_name}\" not in available names: {experiment_configurations.keys()}, register it in config.py")
        
        compiler_copies = {name: compiler.copy() for name, compiler in compilers.items()}

        machine_specific_experiment_configuration = {}
        if experiment_name in machine_specific_experiment_configurations.keys():
            machine_specific_experiment_configuration= machine_specific_experiment_configurations[experiment_name]

        experiments.append(experiment_configurations[experiment_name](constants_configuration, compiler_copies, machine_specific_experiment_configuration))
        
    return experiments