from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import matplotlib.pyplot as plt
import sys
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
import argparse
from pprint import pprint

backend = 'genn'
neuron_class = 'IF_cond_exp'
# heidelberg's brainscales seems to like these params
e_rev = 92 #mV
# e_rev = 500.0 #mV

base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    # 'e_rev_I': -e_rev, #mV
    # 'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms

}

base_params['e_rev_I'] = -e_rev
base_params['e_rev_E'] = 0.0

n_neurons = 3
timestep = 1.

pynnx = PyNNAL(backend)
pynnx._sim.setup(timestep=timestep, min_delay=timestep,
                 extra_params={'use_cpu': True})


pre = pynnx.Pop(n_neurons, neuron_class, base_params)
pre2 = pynnx.Pop(n_neurons, neuron_class, base_params)
post = pynnx.Pop(n_neurons, neuron_class, base_params)


w = [[i, j, 1 + i*n_neurons + j, 1] for i in range(n_neurons) for j in range(n_neurons)]
# print(w)
w_in = np.empty((n_neurons, n_neurons))
for r, c, v, d in w:
    w_in[r, c] = v

# print(w_in)
proj = pynnx.Proj(pre, post, 'FromListConnector', None, None,
                  conn_params={'conn_list': w})
proj2 = pynnx.Proj(pre2, post, 'FromListConnector', None, None,
                   conn_params={'conn_list': w}, target='inhibitory')

pynnx.run(1)

w_out = pynnx.get_weights(proj)

pynnx.end()

# print(w_out)

print((w_in - w_out).sum())