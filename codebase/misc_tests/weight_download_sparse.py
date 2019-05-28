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

n_pre = 10
n_post = 5
timestep = 1.

pynnx = PyNNAL(backend)
pynnx._sim.setup(timestep=timestep, min_delay=timestep,
                 backend='SingleThreadedCPU'
                 )


pre = pynnx.Pop(n_pre, 'SpikeSourceArray', {'spike_times': [[] for _ in range(n_pre)]})
post = pynnx.Pop(n_post, neuron_class, base_params)

t_plus = 1.0
t_minus = 1.0
a_plus = 0.0
a_minus = 0.0
w_max = 10.0

stdp = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': t_plus,
                   'tau_minus': t_minus,
                   # 'tau_minus': 33.7,
                   },
    },
    'weight_dependence': {
        'name':'AdditiveWeightDependence',
        # 'name':'MultiplicativeWeightDependence',
        'params': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': 0.0,
            'w_max': w_max,
            # 'w_max': (static_w['KC to DN']),
            'A_plus': a_plus, 'A_minus': a_minus,
        },
    }
}

# np.random.seed(1)
w = [[i, j, 1 + i*n_post + j, 1] for i in range(n_pre) for j in range(n_post) if np.random.uniform(0., 1.) <= 0.5]
# print(w)
w_in = np.ones((n_pre, n_post)) * np.nan
for r, c, v, d in w:
    w_in[r, c] = v

print(w_in)
proj = pynnx.Proj(pre, post, 'FromListConnector', None, None,
                  conn_params={'conn_list': w}, stdp=stdp)

pynnx.run(1)

w_out = pynnx.get_weights(proj)

pynnx.end()

print(w_out)

sum_abs_diff = 0.0
for r in range(n_pre):
    for c in range(n_post):
        if np.isnan(w_in[r, c]):
            w_in[r, c] = 1e10
        if np.isnan(w_out[r, c]):
            w_out[r, c] = 1e10

        sum_abs_diff += np.abs(w_in[r, c] - w_out[r, c])
# print((w_in - w_out))
print("sum of abs diff", sum_abs_diff)
# print(w_in.flatten())
print(np.where(w_in == 6.))
print(np.where(w_in == 6.)[0]*n_post + np.where(w_in == 6.)[1] )
print(np.where(w_in.flatten() == 6.))
print(np.where(w_in.flatten() == 6.)[0]//n_post, np.where(w_in.flatten() == 6.)[0]%n_post)
# print(w_in.flatten() - w_out.flatten())
