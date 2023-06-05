from pkgutil import iter_modules
import deepspeed.utils

def list_submodules(module):
    for submodule in iter_modules(module.__path__):
        print(submodule.name)

list_submodules(deepspeed)
