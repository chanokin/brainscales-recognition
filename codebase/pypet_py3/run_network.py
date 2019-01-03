import sys
import Pyro4
import Pyro4.util

sys.excepthook = Pyro4.util.excepthook

neuron_parameters = {
    'v_thresh':   -35.0, 
    'tau_m':       20.,
    'tau_syn_E':   10.0, 
    'e_rev_E':     0., 
    'tau_refrac':  0.1 , 
    'v_reset':    -50.0,  #hdbrgs
    'tau_syn_I':   5., 
    'i_offset':    0.0,
    #ESS - BrainScaleS
    'cm':          0.2,
    'v_rest':     -50.0,
    'e_rev_I':    -100.,
} 

description = {
    'populations': {
        'source': {
            'type': 'SpikeSourceArray',
            'size': 1,
            'params': { 
                'spike_times': [10, 20, 30],
            },
            'record': ['spikes'],
        },
        'destination': {
            'type': 'IF_cond_exp',
            'size': 1,
            'params': neuron_parameters,
            'record': ['spikes']
        }
    },
    'projections':{
        'source': {
            'destination': {
                'conn': 'OneToOneConnector',
                'weights': 0.01,
                'delays': 1.0,
            }
        }
    }
}

pynn_server = Pyro4.Proxy("PYRONAME:spikevo.pynn_server")

pynn_server.set_net('nest', description)
pynn_server.run(100)
recs = pynn_server.get_records()
pynn_server.end()

from pprint import pprint

pprint(recs)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for pop in recs:
    fig = plt.figure()
    for i, train in enumerate(recs[pop]['spikes']):
        plt.plot(train, i*np.ones_like(train), '.')
    plt.savefig('{}_spikes.pdf'.format(pop))
