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

pynnx = PyNNAL(pynn, max_subpop_size=2)

N_NEURONS = 10
w = 0.025
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

inputs = pynnx.Pop(N_NEURONS, pynn.SpikeSourcePoisson,
                   {'rate': [10.0] * N_NEURONS},
                   label='poisson'
                   )
inputs.record('spikes')

proj = pynnx.Proj(inputs, neurons, pynn.FixedProbabilityConnector,
                  weights=w, delays=syn_delay,
                  conn_params={
                      'p_connect': 0.1,
                      'rng': pynnx.NumpyRNG()})
# proj = pynnx.Proj(inputs, neurons, pynn.AllToAllConnector,
#                   weights=w, delays=syn_delay)
# proj = pynnx.Proj(inputs, neurons, pynn.OneToOneConnector,
#                   weights=w, delays=syn_delay)

graph = pynnx._graph.clone()
pprint(graph)

pynnx.run(sim_time)

out_spikes = pynnx.get_spikes(neurons)
in_spikes = pynnx.get_spikes(inputs)

if backend != GENN:
    weights = pynnx.get_weights(proj)
else:
    weights = np.zeros((N_NEURONS, N_NEURONS))

pynnx.end()

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
for nid, times in enumerate(in_spikes):
    plt.plot(times, np.ones_like(times) * nid, '+g', markersize=5)
for nid, times in enumerate(out_spikes):
    plt.plot(times, np.ones_like(times) * nid, 'xb', markersize=5)

ax.set_xlim(0, sim_time)
ax.set_ylim(0, N_NEURONS)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron id')

# ax = plt.subplot(1, 2, 2)
# plt.imshow(weights)
# ax.set_xlabel('Post id')
# ax.set_ylabel('Pre id')

plt.savefig("output.pdf")
plt.show()
#
#
#
#
