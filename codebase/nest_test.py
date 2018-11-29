import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pyNN.nest as pynn

N_NEURONS = 50

default_parameters = {
    'tau_refrac':  0.1,
    'cm':          1.0,
    'tau_syn_E':   5.0,
    'v_rest':     -65.0,
    'tau_syn_I':   5.0,
    'tau_m':       20.0,
    'e_rev_E':     0.0,
    'i_offset':    0.0,
    'e_rev_I':    -70.0,
    'v_thresh':   -50.0,
    'v_reset':    -65.0,
}

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

pynn.setup()

neurons = pynn.Population(N_NEURONS, pynn.IF_cond_exp, 
            {'cm': 0.2,
             'e_rev_I': -100.0,
             'v_rest': -50.0,}
          )
neurons.record()
inputs = pynn.Population(N_NEURONS, pynn.SpikeSourcePoisson, 
            {'rate': 10.0}
         )

proj = pynn.Projection(inputs, neurons,
        pynn.OneToOneConnector(weights=0.001))

pynn.run(1000)

out_spikes = np.array(neurons.getSpikes())

pynn.end()

plt.figure()
plt.plot(out_spikes[:, 1], out_spikes[:, 0], '.', markersize=1)
plt.savefig("output.pdf")
plt.show()




