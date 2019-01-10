from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import *
import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from spikevo import *
from spikevo.pynn_transforms import PyNNAL




    
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--backend', help='available %s'%supported_backends, 
    default=GENN)

args = parser.parse_args()
backend = args.backend
pynn = backend_setup(backend)

print(pynn.__name__)

pynnx = PyNNAL(pynn)


N_NEURONS = 49
w = 0.025
syn_delay = 1.
sim_time = 1000.

# default_parameters = {
    # 'tau_refrac':  0.1,
    # 'cm':          1.0,
    # 'tau_syn_E':   5.0,
    # 'v_rest':     -65.0,
    # 'tau_syn_I':   5.0,
    # 'tau_m':       20.0,
    # 'e_rev_E':     0.0,
    # 'i_offset':    0.0,
    # 'e_rev_I':    -70.0,
    # 'v_thresh':   -50.0,
    # 'v_reset':    -65.0,
# }

neuron_parameters = {
    'v_thresh':   -35.0, 
    'tau_m':       20.,
    'tau_syn_E':   10.0, 
    # 'e_rev_E':     0., 
    'tau_refrac':  0.1 , 
    'v_reset':    -50.0,  #hdbrgs
    'tau_syn_I':   5., 
    'i_offset':    0.0,
    #ESS - BrainScaleS
    'cm':          0.2,
    'v_rest':     -50.0,
    # 'e_rev_I':    -100.,
} 

pynn.setup(timestep=1.0, min_delay=1.0)

neurons = pynnx.Pop(N_NEURONS, pynn.IF_cond_exp, neuron_parameters)
neurons.record('spikes')

inputs = pynnx.Pop(N_NEURONS, pynn.SpikeSourcePoisson,
            {'rate': [10.0]*N_NEURONS},
         )
inputs.record('spikes')

proj = pynnx.Proj(inputs, neurons, pynn.OneToOneConnector, 
        weights=w, delays=syn_delay)

pynn.run(sim_time)

out_spikes = pynnx.get_spikes(neurons)
in_spikes = pynnx.get_spikes(inputs)

if backend != GENN:
    weights = pynnx.get_weights(proj)
else:
    weights = np.zeros((N_NEURONS, N_NEURONS))

pynn.end()

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
for nid, times in enumerate(in_spikes):
    plt.plot(times, np.ones_like(times)*nid, '+g', markersize=5)
for nid, times in enumerate(out_spikes):
    plt.plot(times, np.ones_like(times)*nid, 'xb', markersize=5)

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




