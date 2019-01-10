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
from args_setup import get_args
from input_utils import *

e_rev = 80
neuron_parameters = {
    'cm': 0.2,
    'v_reset': -75.,
    'v_rest': -70.,
    'v_thresh': -20.,
    'e_rev_I': -e_rev,
    'e_rev_E': 0.,#e_rev,
    'tau_m': 10.,
    'tau_refrac': 1.0,
    'tau_syn_E': 5.,
    'tau_syn_I': 5.,
}

pynnx = PyNNAL(GENN)
pynnx.setup(timestep=1, min_delay=1, per_sim_params={'use_cpu': True})

texc = 5
tinh = texc + 5
exc = pynnx.Pop(1, 'SpikeSourceArray', {'spike_times': [[texc]]})
inh = pynnx.Pop(1, 'SpikeSourceArray', {'spike_times': [[tinh]]})

dst = pynnx.Pop(1, 'IF_cond_exp', neuron_parameters)
pynnx.set_recording(dst, 'spikes')
pynnx.set_recording(dst, 'v')

w = 0.1122
e2d = pynnx.Proj(exc, dst, 'OneToOneConnector', w, 1.0, target='excitatory')
i2d = pynnx.Proj(inh, dst, 'OneToOneConnector', w, 1.0, target='inhibitory')

pynnx.run(50)

v = pynnx.get_record(dst, 'v')[0]
spikes = pynnx.get_record(dst, 'spikes')

pynnx.end()

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.axvline(texc+1, color='green')
plt.axvline(tinh+1, color='red')
plt.plot(v, color='blue')
print(v)
print(spikes)
plt.show()
