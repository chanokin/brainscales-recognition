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

parser = argparse.ArgumentParser()
parser.add_argument('dc', type=float, help='DC current offset' )

args = parser.parse_args()

e_rev = 92 #mV
neuron_parameters = {
    'cm': 0.09, #nF
    'v_reset': -70., #mV
    'v_rest': -65., #mV
    'v_thresh': -55., #mV
    'e_rev_I': -e_rev, #mV
    'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10., #ms
    'tau_refrac': 5.0, #ms
}

pynnx = PyNNAL(GENN)
pynnx.setup(timestep=1, min_delay=1, per_sim_params={'use_cpu': True})

dst = pynnx.Pop(1, 'IF_cond_exp', neuron_parameters)
pynnx.set_recording(dst, 'spikes')
pynnx.set_recording(dst, 'v')

pulse = pynnx.sim.DCSource(amplitude=args.dc, start=30.0, stop=130.0)
pulse.inject_into(dst)

pynnx.run(200)

v = pynnx.get_record(dst, 'v')[0]
spikes = pynnx.get_record(dst, 'spikes')

pynnx.end()

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.plot(v, color='blue')
ax.set_ylabel('voltage [mV]')
ax.set_xlabel('time [ms]')
ax.set_title('{} spikes'.format(len(spikes[0])))

i_txt = '{}'.format(args.dc).replace('.', 'p')
plt.savefig('response_{}.pdf'.format(i_txt))

plt.show()

np.savez_compressed('response_{}.npz'.format(i_txt),
                    voltage=v, spikes=spikes, dc=args.dc)