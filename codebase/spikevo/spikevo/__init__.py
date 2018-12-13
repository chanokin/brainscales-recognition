import sys
import os
GENN = 'genn'
NEST = 'nest'
BSS  = 'brainscales'
BSS_BACK = 'pyhmf'
supported_backends = [GENN, NEST, BSS]

def backend_setup(backend):
    if backend not in supported_backends:
        raise Exception("Backend not supported")

    if backend == GENN:
        import pynn_genn as pynn_local

    elif backend == NEST:
        sys.path.insert(0, os.environ['PYNEST222_PATH'])
        sys.path.insert(0, os.environ['PYNN7_PATH'])
        import pyNN.nest as pynn_local
        
    elif backend == BSS:
        import pyhmf as pynn_local
    
    return pynn_local


