import numpy as np
import os
from utils import *

SIM_NAME = 'genn'
KERNEL_W = 7
N_INPUT_LAYERS = 4
PAD = KERNEL_W//2
WEIGHT_RANGE = (0.001, 0.1)
PI_DIVS_RANGE = (2, 7)
STRIDE_RANGE = (1, KERNEL_W//2 + 1)
OMEGA_RANGE = (0.1, 3.0)
EXPANSION_RANGE = (10, 51)
EXP_PROB_RANGE = (0.05, 0.5)
OUTPUT_PROB_RANGE = (0.05, 0.5)

ATTRS = ['weight', 'n_pi_divs', 'stride',
    'omega', 'expand', 'exp_prob', 'out_prob']

N_ATTRS = len(ATTRS)

ATTR2IDX = {attr: i for i, attr in enumerate(ATTRS)}

ATTR_RANGES = {
    'weight': WEIGHT_RANGE,
    'n_pi_divs': PI_DIVS_RANGE,
    'stride': STRIDE_RANGE,
    'omega': OMEGA_RANGE,
    'expand': EXPANSION_RANGE,
    'exp_prob': EXP_PROB_RANGE,
    'out_prob': OUTPUT_PROB_RANGE,
}






### Main scale
W2S = 0.0025

### Input generation constants

num_patterns_AL = 10
num_samples_AL = 1000
randomize_samples_AL = True # change order of input samples
prob_active_AL = 0.2
prob_noise_per_sample_AL = 0.1

sample_t_window = 50 #ms
start_dt = 25 #ms
max_rand_dt = 1 #ms

### Connectivity constants
prob_antenna_to_kenyon = 0.15
prob_kenyon_to_decision = 0.2 #how many weights will be high
inactive_k2d_scale = 0.1 #multiply high by this to get low weights

### Neuron types
neuron_class = 'IF_curr_exp'
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

