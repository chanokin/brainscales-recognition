from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--backend', help='available %s' % supported_backends,
                    default=GENN)

args = parser.parse_args()
backend = args.backend
pynn = backend_setup(backend)

print(pynn.__name__)

pynnx = PyNNAL(pynn, max_subpop_size=np.inf)

N_NEURONS = 10
w = 0.02
syn_delay = 1.
sim_time = 1000.

neuron_parameters = {
    'v_thresh': -35.0,
    'tau_m': 20.,
    'tau_syn_E': 10.0,
    # 'e_rev_E':     0.,
    'tau_refrac': 0.1,
    'v_reset': -50.0,  # hdbrgs
    'tau_syn_I': 5.,
    'i_offset': 0.0,
    # ESS - BrainScaleS
    'cm': 0.2,
    'v_rest': -50.0,
    # 'e_rev_I':    -100.,
}

pynn.setup(timestep=1.0, min_delay=1.0)

neurons = pynnx.Pop(N_NEURONS, pynn.IF_cond_exp, neuron_parameters,
                    label='neurons')
neurons.record('spikes')

output  = pynnx.Pop(N_NEURONS, pynn.IF_cond_exp, neuron_parameters,
                    label='outputs')
output.record('spikes')

inputs = pynnx.Pop(N_NEURONS, pynn.SpikeSourcePoisson,
            {'rate': [10.0] * N_NEURONS}, label='poisson')
inputs.record('spikes')

inputs1 = pynnx.Pop(N_NEURONS, pynn.SpikeSourcePoisson,
            {'rate': [10.0] * N_NEURONS}, label='poisson new')
inputs1.record('spikes')


proj = pynnx.Proj(inputs, neurons, pynn.FixedProbabilityConnector,
        weights=w, delays=syn_delay,
        conn_params={'p_connect': 0.1, 'rng': pynnx.NumpyRNG(),})

proj1 = pynnx.Proj(inputs, output, pynn.FixedProbabilityConnector,
        weights=w, delays=syn_delay,
        conn_params={'p_connect': 0.1, 'rng': pynnx.NumpyRNG(),})

proj2 = pynnx.Proj(neurons, output, pynn.FixedProbabilityConnector,
            weights=w, delays=syn_delay,
            conn_params={'p_connect': 0.1, 'rng': pynnx.NumpyRNG(),})


proj3 = pynnx.Proj(inputs1, output, pynn.FixedProbabilityConnector,
            weights=w, delays=syn_delay,
            conn_params={'p_connect': 0.1, 'rng': pynnx.NumpyRNG(),})

# proj = pynnx.Proj(inputs, neurons, pynn.AllToAllConnector,
#                   weights=w, delays=syn_delay)
# proj = pynnx.Proj(inputs, neurons, pynn.OneToOneConnector,
#                   weights=w, delays=syn_delay)


pops = ['neurons', 'outputs']
sources = ['poisson', 'poisson new']
targets = {'neurons': ['outputs'],
           'poisson': ['neurons', 'outputs'],
           'poisson new': ['outputs']
           }



graph = pynnx._graph.clone()

all_pops_in = True
for p in pops:
    all_pops_in &= (p in graph.nodes)

assert all_pops_in, "missing standard population in the graph"

all_sources_in = True
for s in sources:
    all_sources_in &= (s in graph.sources)

assert all_sources_in, "missing source population in the graph"

all_conns_in = True
for pre in targets:
    for post in targets[pre]:
        if pre in graph.nodes:
            all_conns_in &= (post in graph.nodes[pre].outputs)
        else:
            all_conns_in &= (post in graph.sources[pre].outputs)

assert all_conns_in, "missing connections in the graph"