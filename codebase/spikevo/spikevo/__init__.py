from __future__ import (print_function,
                        unicode_literals,
                        division)
import sys
import os
import argparse

GENN = 'genn'
NEST = 'nest'
BSS  = 'brainscales'
BSS_BACK = 'pyhmf'
supported_backends = [GENN, NEST, BSS]

RED, GREEN, BLUE = range(3)
ON, OFF = range(2)
CHAN2COLOR = {ON: GREEN, OFF: RED}
CHAN2TXT = {ON: 'GREEN', OFF: 'RED'}
RATE = 'rate'
ON_OFF = 'on-off'
IMAGE_ENCODINGS = [RATE, ON_OFF]
MAX_SUBPOP_SIZE = 150

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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_n_part(size, part_size):
    return size//part_size + int(size % part_size > 0)
