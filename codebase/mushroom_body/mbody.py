from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict


import numpy as np
import matplotlib.pyplot as plt

from spikevo import *
from spikevo.pynn_transforms import PyNNAL
import argparse
from pprint import pprint
from args_setup import get_args

args = get_args()
pprint(args)

backend = GENN

#heidelberg's brainscales seems to like these params
e_rev = 50
neuron_parameters = {
    'cm': 0.2,
    'v_reset': -70.,
    'v_rest': -50.,
    'v_thresh': -25.,
    'e_rev_I': -e_rev,
    'e_rev_E': e_rev,
    'tau_m': 10.,
    'tau_refrac': 0.1,
    'tau_syn_E': 5.,
    'tau_syn_I': 5.,
}


def generate_input_vectors(num_vectors, dimension, on_probability, seed=1):
    np.random.seed(seed)
    vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    np.random.seed()
    return vecs

def generate_samples(input_vectors, num_samples, prob_noise, seed=1):
    np.random.seed(seed)
    
    samples = None
    for i in range(input_vecs.shape[0]):
        samp = np.tile(input_vecs[i, :], (num_samples, 1)).astype('int')
        dice = np.random.uniform(0., 1., samp.shape)
        whr = np.where(dice < prob_noise)
        samp[whr] = 1 - samp[whr]
        if samples is None:
            samples = samp
        else:
            samples = np.append(samples, samp, axis=0)

    np.random.seed()
    return samples

def samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, seed=1):
    np.random.seed(seed)
    t = 0
    spike_times = [[] for _ in range(samples.shape[-1])]
    rand_idx = np.random.choice(np.arange(samples.shape[0]), size=samples.shape[0],
                replace=False)
    rand_idx = np.arange(samples.shape[0])
    for idx in rand_idx:
        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        ts = t + start_dt + np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size) 
        for time_id, neuron_id in enumerate(active):
            spike_times[neuron_id].append(ts[time_id])

        t += sample_dt
    np.random.seed()
    return spike_times

# input_vecs = generate_input_vectors(args.nPatternsAL, args.nAL, args.probAL)
input_vecs = generate_input_vectors(10, 100, 0.1)
pprint(input_vecs)

samples = generate_samples(input_vecs, 100, 0.01)
pprint(samples)

sample_dt, start_dt, max_rand_dt = 50, 5, 2
spike_times = samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt)
plt.figure()
for i, times in enumerate(spike_times):
    plt.plot(times, np.ones_like(times)*i, '.b')
plt.show()


