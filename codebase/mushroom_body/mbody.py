from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict


import numpy as np
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
import argparse
from pprint import pprint
from args_setup import get_args

args = get_args()
pprint(args)

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
    
    samples = []
    for i in range(input_vecs.shape[0]):
        samp = np.tile(input_vecs[i, :], (1, num_samples, 1))
        dice = np.random.uniform(0., 1., samp.shape)
        samp[dice < prob_noise] ^= samp[dice < prob_noise] # xor to flip current values
        samples.append(samp)

    np.random.seed()
    return samples


def partition_population(size, max_pop_size=175, structure=None):
    pass


# input_vecs = generate_input_vectors(args.nPatternsAL, args.nAL, args.probAL)
input_vecs = generate_input_vectors(3, 3, 0.2)
pprint(input_vecs)

samples = generate_samples(input_vecs, 10, 0.5)
pprint(samples)
