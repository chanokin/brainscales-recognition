import numpy as np
import os
from utils import *

multiproc = True
n_procs = 4

SIM_NAME = 'genn'
DEBUG = bool(1)

N_CLASSES = 1 if DEBUG else 10
N_SAMPLES = 1 if DEBUG else 100
KERNEL_W = 7
N_INPUT_LAYERS = 4
PAD = KERNEL_W//2
PI_DIVS_RANGE = (1, 2) if DEBUG else (2, 7)
STRIDE_RANGE = (3, 4) if DEBUG else (1, KERNEL_W//2 + 1)
OMEGA_RANGE = (0.5, 3.0)
EXPANSION_RANGE = (1, 2) if DEBUG else (2, 6)
EXP_PROB_RANGE = (0.01, 0.011) if DEBUG else (0.05, 0.5)
OUTPUT_PROB_RANGE = (0.01, 0.011) if DEBUG else (0.05, 0.5)
OUTPUT_SIZE = N_CLASSES * 10
OUT_WEIGHT_RANGE = (4.0, 5.0)

ATTRS = ['out_weight', 'n_pi_divs', 'stride',
    'omega', 'expand', 'exp_prob', 'out_prob']

N_ATTRS = len(ATTRS)

ATTR2IDX = {attr: i for i, attr in enumerate(ATTRS)}

ATTR_RANGES = {
    'out_weight': OUT_WEIGHT_RANGE,
    'n_pi_divs': PI_DIVS_RANGE,
    'stride': STRIDE_RANGE,
    'omega': OMEGA_RANGE,
    'expand': EXPANSION_RANGE,
    'exp_prob': EXP_PROB_RANGE,
    'out_prob': OUTPUT_PROB_RANGE,
}

TIMESTEP = 0.1
SAMPLE_DT = 50.0#ms



### Main scale
W2S = 0.0025


### Neuron types
neuron_class = 'IF_curr_exp'
gabor_class = 'IF_curr_exp'
mushroom_class = 'IF_curr_exp'
output_class = 'IF_curr_exp'

neuron_types = {
    'antenna': 'SpikeSourceArray',
    'kenyon': neuron_class,
    'horn': neuron_class,
    'decision': neuron_class,
}

### Population sizes
pop_sizes = {
    'antenna': 100,
    'kenyon': 2500,
    'horn': 20,
    'decision': 100,
}
### Neuron configuration
base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
}

gabor_params = base_params.copy()
mushroom_params = base_params.copy()
output_params = base_params.copy()

gabor_weight = [10.0, 10.0, 50.0, 50.0]
mushroom_weight = 1.0

record_spikes = ['input', 'gabor', 'mushroom', 'output']
record_weights = ['input to gabor', 'gabor to mushroom', 'mushroom to output']
# record_weights = ['gabor to mushroom', 'mushroom to output']
time_dep = 'SpikePairRule'
tau_plus = 20.0
tau_minus = 20.0
A_plus = 0.01
A_minus = 0.01

weight_dep = 'AdditiveWeightDependence'
w_min_mult = 0.0
w_max_mult = 1.2