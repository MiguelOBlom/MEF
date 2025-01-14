import re
import copy
import subprocess

class Compiler:
    def __init__(self, cc, mode='', standard='', debug=False, profile=False, opt='', warn=[], pedantic=False, include=[], lib=[], dmacro={}, umacro='', fopt=[], mopt=[], outfile='', flagfile='', infiles=[], libraries=[]):
        self.cc = cc
        self.mode = mode
        self.standard = standard
        self.debug = debug
        self.profile = profile
        self.opt = opt
        self.warn = warn
        self.pedantic = pedantic
        self.include = include
        self.lib = lib
        self.dmacro = dmacro
        self.umacro = umacro
        self.fopt = fopt
        self.mopt = mopt
        self.outfile = outfile
        self.flagfile = flagfile
        self.infiles = infiles
        self.libraries = libraries

    def prebuild(self, infile, outfile):
        pass

    def compile(self, infile, outfile):
        # Build compilation command
        cc = self.cc
        mode = (f"-{self.mode}" if self.mode[0] != '-' else self.mode) if self.mode else ''
        standard = (f"-std={self.standard}" if self.standard[0] != '-' else self.standard) if self.standard else ''
        debug_flag = "-g -gdwarf-4" if self.debug else ''
        profile = '-pg' if self.profile else ''
        opt = (f"-O{self.opt}" if str(self.opt)[0] != '-' else self.opt) if self.opt else ''
        warn = ' '.join([(f"-W{w}" if w[0] != '-' else w) for w in self.warn])
        warn = warn + ' -Wpedantic' if self.pedantic else warn
        include = ' '.join([(f"-I{i}" if i[0] != '-' else i) for i in self.include])
        lib = ' '.join([(f"-L{l}" if l[0] != '-' else l) for l in self.lib])
        dmacro = ' '.join([(f"-D{k}={v}" if v != '' else f"-D{k}") if k[0] != '-' else (f"{k}={v}" if v else k) for k,v in self.dmacro.items()])
        umacro = (f"-U{self.umacro}" if self.umacro[0] != '-' else self.umacro) if self.umacro else ''
        fopt = ' '.join([(f"-f{f}" if f[0] != '-' else f) for f in self.fopt])
        mopt = ' '.join([(f"-m{m}" if m[0] != '-' else m) for m in self.mopt])
        flagfile = (f"@{self.flagfile}" if self.flagfile[0] != '@' else self.flagfile) if self.flagfile else ''
        libraries = ' '.join([(f"-l{l}" if l[0] != '-' else l) for l in self.libraries])

        infiles_str = ' '.join([infile] + self.infiles)
        outfile_str = f"-o {outfile}"

        command = f"{cc} {mode} {standard} {debug_flag} {profile} {opt} {warn} {include} {lib} {dmacro} {umacro} {fopt} {mopt} {outfile_str} {flagfile} {infiles_str} {libraries}"
        command = re.sub(' +', ' ', command).strip()

        print(command)
        subprocess.call(command.split(' '))

    def copy(self, cc=None, mode=None, standard=None, debug=None, 
             profile=None, opt=None, warn=[], pedantic=None, 
             include=[], lib=[], dmacro={}, umacro=None, 
             fopt=[], mopt=[], outfile=None, flagfile=None, 
             infiles=[], libraries=[]):
        compiler = copy.deepcopy(self)

        compiler.cc = cc if not cc is None else compiler.cc
        compiler.mode = mode if not mode is None else compiler.mode
        compiler.standard = standard if not standard is None else compiler.standard
        compiler.debug = debug if not debug is None else compiler.debug
        compiler.profile = profile if not profile is None else compiler.profile
        compiler.opt = opt if not opt is None else compiler.opt
        compiler.warn = list(set(compiler.warn + warn))
        compiler.pedantic = pedantic if not pedantic is None else compiler.pedantic
        compiler.include = list(set(compiler.include + include))
        compiler.lib = list(set(compiler.lib + lib))
        compiler.dmacro.update(dmacro)
        compiler.umacro = umacro if not umacro is None else compiler.umacro
        compiler.fopt = list(set(compiler.fopt + fopt))
        compiler.mopt = list(set(compiler.mopt + mopt))
        compiler.outfile = outfile if not outfile is None else compiler.outfile
        compiler.flagfile = flagfile if not flagfile is None else compiler.flagfile
        compiler.infiles = list(set(compiler.infiles + infiles))
        compiler.libraries = list(set(compiler.libraries + libraries))

        return compiler
    
    def update_from_compiler(self, compiler):
        self.cc = compiler.cc if not compiler.cc is None else self.cc
        self.mode = compiler.mode if not compiler.mode is None else self.mode
        self.standard = compiler.standard if not compiler.standard is None else self.standard
        self.debug = compiler.debug if not compiler.debug is None else self.debug
        self.profile = compiler.profile if not compiler.profile is None else self.profile
        self.opt = compiler.opt if not compiler.opt is None else self.opt
        self.warn = list(set(self.warn + compiler.warn))
        self.pedantic = compiler.pedantic if not compiler.pedantic is None else self.pedantic
        self.include = list(set(self.include + compiler.include))
        self.lib = list(set(self.lib + compiler.lib))
        self.dmacro.update(compiler.dmacro)
        self.umacro = compiler.umacro if not compiler.umacro is None else self.umacro
        self.fopt = list(set(self.fopt + compiler.fopt))
        self.mopt = list(set(self.mopt + compiler.mopt))
        self.outfile = compiler.outfile if not compiler.outfile is None else self.outfile
        self.flagfile = compiler.flagfile if not compiler.flagfile is None else self.flagfile
        self.infiles = list(set(self.infiles + compiler.infiles))
        self.libraries = list(set(self.libraries + compiler.libraries))