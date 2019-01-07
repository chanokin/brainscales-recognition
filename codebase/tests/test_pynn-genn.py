import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pynn_genn as pynn

N_NEURONS = 50
w = 0.01
syn_delay = 1.

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

pynn.setup(timestep=1.0, use_cpu=True)

neurons = pynn.Population(N_NEURONS, 
            pynn.IF_cond_exp(**neuron_parameters),
          )
neurons.record('spikes')

inputs = pynn.Population(N_NEURONS, 
            pynn.SpikeSourcePoisson(rate=10.0),
         )

syn = pynn.StaticSynapse(weight=w, delay=syn_delay)
proj = pynn.Projection(inputs, neurons,
        pynn.OneToOneConnector(), syn)

pynn.run(1000.0)

data = neurons.get_data()

if len(data.segments):
    data = data.segments[0]
    out_spikes = np.array(data.spiketrains)

    pynn.end()

    plt.figure()
    for nid, times in enumerate(out_spikes):
        plt.plot(times, np.ones_like(times)*nid, '.b', markersize=1)
    plt.savefig("output.pdf")
    plt.show()




