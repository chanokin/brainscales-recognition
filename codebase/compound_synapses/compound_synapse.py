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

timestep = 0.1
max_w = 0.01
start_w = max_w / 2.0

tau_plus = 10.0
tau_minus = 50.0
a_plus = 0.01
a_minus = 0.005
delays = range(1, 11)

start_dt, num_dt, ddt = -150, 300, 1
sim_time = np.round(1.5 * num_dt)
start_t = sim_time - num_dt
trigger_t = start_t + (start_dt + num_dt//2)
num_neurons = num_dt

delays = [1, 3]
dt_dw = {d: {} for d in delays}
out_w  = {d: {} for d in delays}
pynnx = PyNNAL(backend)
pynnx._sim.setup(timestep=timestep, min_delay=timestep,
                 extra_params={'use_cpu': True})


projs = {d: {} for d in delays}
for dt in range(start_dt, start_dt+num_dt, ddt):
    pre_spike_times = [[trigger_t + dt]]
    trigger_spike_times = [[trigger_t]]

    trigger = pynnx.Pop(1, 'SpikeSourceArray',
                        {'spike_times': trigger_spike_times},
                        label='trigger')

    post = pynnx.Pop(1, neuron_class, base_params)
    pynnx.set_recording(post, 'spikes')

    pre = pynnx.Pop(1, 'SpikeSourceArray',
                    {'spike_times': pre_spike_times},
                    label='delay = 1')
    tr2post = pynnx.Proj(trigger, post, 'OneToOneConnector', 0.1, 1.0, label='trigger connection')


    stdp = {
        'timing_dependence': {
            'name': 'SpikePairRule',
            'params': {'tau_plus': tau_plus,
                       'tau_minus': tau_minus,
                       # 'tau_minus': 33.7,
                       },
        },
        'weight_dependence': {
            'name':'AdditiveWeightDependence',
            # 'name':'MultiplicativeWeightDependence',
            'params': {
                # 'w_min': (static_w['KC to DN'])/10.0,
                'w_min': 0.0,
                'w_max': max_w,
                # 'w_max': (static_w['KC to DN']),
                'A_plus': a_plus,
                'A_minus': a_minus,
                # 'A_plus': max_w * a_plus,
                # 'A_minus': max_w * a_minus,
            },
        }
    }

    pre2post = pynnx.Proj(pre, post, 'AllToAllConnector', start_w, delays[0],
                          stdp=stdp, label='plastic connection delay = 1')

    projs[delays[0]][dt] = pre2post

    ##################### --------------------- ######################

    stdp = {
        'timing_dependence': {
            'name': 'SpikePairRule',
            'params': {'tau_plus': tau_minus,
                       'tau_minus': tau_plus,
                       # 'tau_minus': 33.7,
                       },
        },
        'weight_dependence': {
            'name':'AdditiveWeightDependence',
            # 'name':'MultiplicativeWeightDependence',
            'params': {
                # 'w_min': (static_w['KC to DN'])/10.0,
                'w_min': 0.0,
                'w_max': max_w,
                # 'w_max': (static_w['KC to DN']),
                'A_plus': -a_minus,
                'A_minus': -a_plus,
                # 'A_plus': max_w * a_plus,
                # 'A_minus': max_w * a_minus,
            },
        }
    }

    pre2post = pynnx.Proj(pre, post, 'AllToAllConnector', start_w, 1.0,
                          # delays[1],
                          stdp=stdp, label='plastic connection delay = 5')

    projs[delays[1]][dt] = pre2post


pynnx.run(sim_time)

for d in projs:
    for dt in projs[d]:
        out_w[d][dt] = pynnx.get_weights(projs[d][dt])[0,0]
        dt_dw[d][dt] = (out_w[d][dt] - start_w) / max_w

pynnx.end()

total_dt_dw = {}
for d in dt_dw:
    for dt in dt_dw[d]:
        dw = total_dt_dw.get(dt, 0.0)
        dw += dt_dw[d][dt]
        total_dt_dw[dt] = dw

np.savez_compressed('compound_synapse.npz', total_dt_dw=total_dt_dw,
                    dt_dw=dt_dw, end_w=out_w)

plt.figure()
ax = plt.subplot()
plt.axvline(0, linestyle='--', color='gray')
plt.axhline(0, linestyle='--', color='gray')

dts = sorted(total_dt_dw.keys())
dws = [total_dt_dw[dt] for dt in dts]
plt.plot(dts, dws)

max_dw = np.max(np.abs(dws)) * 1.5
ax.set_ylim(-max_dw, max_dw)
ax.set_xlabel(r'$\Delta t = t_{pre} - t_{post}$ [ms]')
ax.set_ylabel(r'$\Delta w $')
plt.legend()
plt.grid()
plt.show()

