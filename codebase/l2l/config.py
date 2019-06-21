import numpy as np
import os
from utils import *

multiproc = True
n_procs = 4

SIM_NAME = 'genn'
DEBUG = bool(1)

TIMESTEP = 0.1 #ms
SAMPLE_DT = 50.0 #ms
N_CLASSES = 2 if DEBUG else 10
N_SAMPLES = 1 if DEBUG else 100
N_TEST = 1 if DEBUG else 10
TOTAL_SAMPLES = N_SAMPLES + N_TEST
KERNEL_W = 7
N_INPUT_LAYERS = 4
PAD = KERNEL_W//2
PI_DIVS_RANGE = (6, 7) if DEBUG else (2, 7)
STRIDE_RANGE = (2, 3) if DEBUG else (1, KERNEL_W//2 + 1)
OMEGA_RANGE = (0.5, 1.0)
EXPANSION_RANGE = (1, 2) if DEBUG else (2, 6)
EXP_PROB_RANGE = (0.05, 0.0500001) if DEBUG else (0.05, 0.5)
OUTPUT_PROB_RANGE = (0.05, 0.0500001) if DEBUG else (0.05, 0.5)
OUTPUT_SIZE = N_CLASSES * 10
OUT_WEIGHT_RANGE = (0.2, 0.200001) if DEBUG else (0.01, 0.5)
GABOR_WEIGHT_RANGE = (2.0, 2.000001) if DEBUG else (1.0, 5.0)
MUSHROOM_WEIGHT_RANGE = (0.25, 0.2500001) if DEBUG else (0.1, 1.0)

### static weights
# gabor_weight = [1.0, 1.0, 2.0, 2.0]
# mushroom_weight = 0.25
inhibitory_weight = -5.0
excitatory_weight = {
    'gabor': 3.0,
    'mushroom': 0.5,
    'output': 3.0,
}

ATTRS = ['out_weight', 'n_pi_divs', 'stride',
    'omega', 'expand', 'exp_prob', 'out_prob',
    'mushroom_weight']
ATTRS += ['gabor_weight-%d'%i for i in range(N_INPUT_LAYERS)]

N_ATTRS = len(ATTRS)

ATTR2IDX = {attr: i for i, attr in enumerate(ATTRS)}

ATTR_RANGES = {
    'out_weight': OUT_WEIGHT_RANGE,
    'mushroom_weight': MUSHROOM_WEIGHT_RANGE,
    'n_pi_divs': PI_DIVS_RANGE,
    'stride': STRIDE_RANGE,
    'omega': OMEGA_RANGE,
    'expand': EXPANSION_RANGE,
    'exp_prob': EXP_PROB_RANGE,
    'out_prob': OUTPUT_PROB_RANGE,
}
for s in ATTRS:
    if s.startswith('gabor_weight'):
        ATTR_RANGES[s] = GABOR_WEIGHT_RANGE


### Neuron types
neuron_class = 'IF_curr_exp'
gabor_class = 'IF_curr_exp'
mushroom_class = 'IF_curr_exp'
inh_mushroom_class = 'IF_curr_exp'
output_class = 'IF_curr_exp'
inh_output_class = 'IF_curr_exp'

### Neuron configuration
base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
    'tau_syn_E': 1., # ms
    'tau_syn_I': 2., # ms

}

gabor_params = base_params.copy()
mushroom_params = base_params.copy()
inh_mushroom_params = base_params.copy()
output_params = base_params.copy()
inh_output_params = base_params.copy()


record_spikes = ['input', 'gabor', 'mushroom', 'output',
                 'inh_mushroom', 'inh_output',
                 ]
record_weights = ['input to gabor',
                  # 'gabor to mushroom',
                  'mushroom to output'
                  ]
time_dep = 'SpikePairRule'
tau_plus = 20.0
tau_minus = 20.0
A_plus = 0.01
A_minus = 0.01

weight_dep = 'AdditiveWeightDependence'
w_min_mult = 0.0
w_max_mult = 1.2