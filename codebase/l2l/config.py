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
OMEGA_RANGE = (0.1, 3.0)
EXPANSION_RANGE = (1, 2) if DEBUG else (2, 6)
EXP_PROB_RANGE = (0.01, 0.011) if DEBUG else (0.05, 0.5)
OUTPUT_PROB_RANGE = (0.01, 0.011) if DEBUG else (0.05, 0.5)
OUTPUT_SIZE = N_CLASSES * 10
OUT_WEIGHT_RANGE = (0.001, 0.1)

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

gabor_weight = [1.0, 1.0, 5.0, 5.0]
mushroom_weight = 1.0

record_spikes = ['inputs', 'gabor', 'mushroom', 'output']

time_dep = 'SpikePairRule'
tau_plus = 20.0
tau_minus = 20.0
A_plus = 0.01
A_minus = 0.01

weight_dep = 'AdditiveWeightDependence'
w_min_mult = 0.0
w_max_mult = 1.2