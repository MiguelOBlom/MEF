import os
import sys
import config
import subprocess
from classes import Logger
import shutil

REALPATH=os.path.dirname(os.path.realpath(__file__))

def usage():
    print(f"Usage {sys.argv[0]} E/T <machine> <experiment[,...]> [-g]")
    print(f"Usage {sys.argv[0]} P <machine> [<experiment[,...]>]")
    print(f"Usage {sys.argv[0]} U [<machine[,...]>]")
    print(f"Usage {sys.argv[0]} D [<machine[,...]>]")
    print(f"Usage {sys.argv[0]} C [-a]")
    print("Options include: ")
    print("\t-g: Debug mode")
    print("\t-a: Clean all, so also the resources and results directories")
    exit()

def get_machines(get_all):
    if get_all:
        machine_names = config.machine_configs.keys()
    else:
        machine_names = sys.argv[2].split(',')
        
        for machine_name in machine_names:
            if not machine_name in config.machine_configs.keys():
                Logger.fail(f"{machine_name} not in available set of machine names: {config.machine_configs.keys()}")
    return machine_names

def up():
    machine_names = get_machines(len(sys.argv) < 3)
    for machine_name in machine_names:
        if "remote" in config.machine_configs[machine_name][0].keys():
            command = f"rsync --info=progress2 -u -a {os.path.join(REALPATH, '')} {os.path.join(config.machine_configs[machine_name][0]['remote'], os.path.basename(REALPATH), '')}"
            print(command)
            subprocess.call(command.split(' '))

def down():
    machine_names = get_machines(len(sys.argv) < 3)
    for machine_name in machine_names:
        if "remote" in config.machine_configs[machine_name][0].keys():
            command = f"rsync --info=progress2 -u -a {os.path.join(config.machine_configs[machine_name][0]['remote'], os.path.basename(REALPATH), '')} {os.path.join(REALPATH, '')}"
            print(command)
            subprocess.call(command.split(' '))

def exec():
    mode = sys.argv[1]
    if len(sys.argv) < 4:
        usage()

    machine_name = sys.argv[2]
    experiment_names = sys.argv[3].split(',')
    options = sys.argv[4:]

    if machine_name not in config.machine_configs.keys():
        Logger.fail(f"Machine name \"{machine_name}\" not in available names: {config.machine_configs.keys()}")

    configuration_options = {}
    configuration_options["realpath"] = REALPATH
    configuration_options["debug"] = "-g" in options

    experiments = config.configure(machine_name, experiment_names, **configuration_options)

    for experiment in experiments:
        if mode == "T":
            experiment.test()
        else:
            if mode != "P":
                experiment.run()
            experiment.plot()

def clean():
    for _, experiment_class in config.experiment_configurations.items():
        experiment_class.clean(REALPATH)
    
    for root, _, _ in os.walk(REALPATH):
        if os.path.basename(root) == '__pycache__':
            shutil.rmtree(root, ignore_errors=True)    
    
    if "-a" in sys.argv:
        shutil.rmtree(os.path.join(REALPATH, "resources"), ignore_errors=True)
        shutil.rmtree(os.path.join(REALPATH, "results"), ignore_errors=True)
        shutil.rmtree(os.path.join(REALPATH, "__pycache__"), ignore_errors=True)


if len(sys.argv) < 2:
    usage()
mode = sys.argv[1]
if mode == "E" or mode == "T" or mode == "P":
    exec()
elif mode == "U":
    up()
elif mode == "D":
    down()
elif mode == "C":
    clean()
else:
    usage()