from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict
import numpy as np
import os
from utils import *


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
neuron_types = {
    'antenna': 'SpikeSourceArray',
    'kenyon': 'IF_cond_exp',
    'horn': 'IF_cond_exp',
    'decision': 'IF_cond_exp',
}

### Population sizes
pop_sizes = {
    'antenna': 100,
    'kenyon': 2500,
    'horn': 20,
    'decision': 100,
}
### Neuron configuration
e_rev = 92  # mV
base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'e_rev_I': -e_rev,  # mV
    'e_rev_E': 0.,  # e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
}

kenyon_parameters = base_params.copy()
kenyon_parameters['tau_syn_E'] = 1.0  # ms
kenyon_parameters['tau_syn_I'] = 5.0  # ms

horn_parameters = base_params.copy()
horn_parameters['tau_syn_E'] = 1.0  # ms

decision_parameters = base_params.copy()
decision_parameters['tau_syn_E'] = 1.0  # ms
decision_parameters['tau_syn_I'] = 5.0  # ms

if os.path.isfile('./input_data_cache.npz'):
    in_data_cache = np.load('./input_data_cache.npz')
    input_vecs = in_data_cache['input_vecs'].tolist()
    samples = in_data_cache['samples'].tolist()
    sample_indices = in_data_cache['sample_indices'].tolist()
    spike_times = [[float(t) for t in times] for times in in_data_cache['spike_times']]
else:
    input_vecs = generate_input_vectors(num_patterns_AL, pop_sizes['antenna'],
                    prob_active_AL, seed=123)

    samples = generate_samples(input_vecs, num_samples_AL, prob_noise_per_sample_AL,
                   seed=234, method='exact')

    sample_indices, spike_times = samples_to_spike_times(samples, sample_t_window, start_dt,
                                     max_rand_dt, randomize_samples=randomize_samples_AL, seed=345)


    input_vecs = input_vecs.tolist()
    samples = samples.tolist()
    sample_indices = sample_indices.tolist()
    np.savez_compressed('./input_data_cache.npz', input_vecs=input_vecs, samples=samples,
            sample_indices=sample_indices, spike_times=spike_times)

neuron_params = {
    'antenna': {'spike_times': spike_times},
    'kenyon': kenyon_parameters,
    'horn': horn_parameters,
    'decision': decision_parameters,
}

static_w = {
    'AL to KC': W2S*(100.0/float(pop_sizes['antenna'])),
    'AL to LH': W2S*(6.5 * (100.0/float(pop_sizes['antenna']))),
    'LH to KC': W2S*(1.925 * (20.0/float(pop_sizes['horn']))),
    'KC to KC': W2S*(0.1*(2500.0/float(pop_sizes['kenyon']))),
    'KC to DN': W2S*(1.5 * (2500.0/float(pop_sizes['kenyon']))),
    'DN to DN': W2S*(1. * (100.0/float(pop_sizes['decision']))),
}

rand_w = {
    'AL to KC': {
        'type': 'normal',
        'params': [static_w['AL to KC'], static_w['AL to KC']*0.2],
        'seed': 1,
     },
}

if os.path.isfile('./starting_connectivity.npz'):
    starting_connectivity = np.load('./starting_connectivity.npz')
    gain_list = starting_connectivity['gain_list'].tolist()
    out_list = starting_connectivity['out_list'].tolist()
else:
    gain_list = gain_control_list(pop_sizes['antenna'],
                  pop_sizes['horn'], static_w['AL to LH'])

    out_list = output_connection_list(pop_sizes['kenyon'], pop_sizes['decision'],
                  prob_kenyon_to_decision, static_w['KC to DN'], inactive_k2d_scale,
                  seed=None)

    gain_list = gain_list.tolist()
    out_list = out_list.tolist()

    np.savez_compressed('./starting_connectivity.npz',
                        gain_list=gain_list, out_list=out_list)

stdp = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': 16.8, 'tau_minus': 33.7},
    },
    'weight_dependence': {
        'name':'MultiplicativeWeightDependence',
        'params': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': 0.0,
            'w_max': (static_w['KC to DN']),
            'A_plus': 0.01, 'A_minus': 0.05
        },
    }
}