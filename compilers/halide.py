import os
import subprocess
from classes import Compiler


class HalideCompiler(Compiler):
    def __init__(self, cc, root_path, src_dir, resource_dir, minimal_compiler, autoschedulers, mode='', standard='', debug=False, profile=False, opt='', warn=[], pedantic=False, include=[], lib=[], dmacro={}, umacro='', fopt=[], mopt=[], outfile='', flagfile='', infiles=[], libraries=[]):
        super().__init__(cc, mode=mode, standard=standard, debug=debug, profile=profile, opt=opt, warn=warn, pedantic=pedantic, include=include, lib=lib, dmacro=dmacro, umacro=umacro, fopt=fopt, mopt=mopt, outfile=outfile, flagfile=flagfile, infiles=infiles, libraries=libraries)
        self.root_path = root_path
        self.src_dir = src_dir
        self.resource_dir = resource_dir
        self.minimal_compiler = minimal_compiler
        self.autoschedulers = autoschedulers

    def prebuild(self, generators, side, trueN):
        # Prebuild autoschedulers
        kernel_name = generators[0]
        halide_dir = os.path.join(self.resource_dir, kernel_name, trueN)
        os.makedirs(halide_dir, exist_ok=True)

        minimal_compiler = self.minimal_compiler.copy(dmacro={"SIDE": side})
        
        infile = os.path.join(self.src_dir, f"{kernel_name}.cpp")
        outfile = os.path.join(halide_dir, kernel_name)
        minimal_compiler.warn = []
        minimal_compiler.compile(infile, outfile)

        env = os.environ.copy()
        ld_library_path = ''
        if 'LD_LIBRARY_PATH' in env.keys():
            ld_library_path = env["LD_LIBRARY_PATH"] + ':'
        env["LD_LIBRARY_PATH"] = f"{ld_library_path}{os.path.join(self.root_path, 'bin')}"

        includes_infiles = []

        for autoscheduler, autoscheduler_lib in self.autoschedulers:
            autoscheduler_dir = os.path.join(halide_dir, autoscheduler)
            includes_infiles.append(([autoscheduler_dir], [os.path.join(autoscheduler_dir, f"{generator}halide.a") for generator in generators]))
            
            if not os.path.exists(autoscheduler_dir):
                os.makedirs(autoscheduler_dir, exist_ok=True)
                for generator in generators:
                    gen_cmd = f"{outfile} -o {autoscheduler_dir} -g {generator}_auto_schedule_gen -f {generator}halide -e static_library,h,schedule -p {os.path.join(self.root_path, 'bin', autoscheduler_lib)} target=x86-64-linux-avx-avx2-f16c-fma-sse41 autoscheduler={autoscheduler} autoscheduler.parallelism=1"
                    print(gen_cmd)
                    subprocess.run(gen_cmd.split(' '), env=env, check=True)

        self.warn = []
        return includes_infiles
