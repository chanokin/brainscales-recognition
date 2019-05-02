import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pynn_genn as pynn
from pyNN.utility.plotting import *

def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=True)


def plot_signal(signal, index, colour=None):
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], color=colour, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    plt.legend()


N_NEURONS = 1
w = 0.005
sim_timestep = 0.1
syn_delay = sim_timestep
v_init = -50.0
v_thr = -35.0

neuron_parameters = {
    'v_thresh':   v_thr,
    'tau_m':       20.,
    'tau_refrac':  10.0,
    'v_reset':    -60.0,  #hdbrgs
    'tau_syn_E':   5.0,
    'tau_syn_I':   5.0,
    'i_offset':    0.0,
    #ESS - BrainScaleS
    'cm':          0.2,
    'v_rest':     v_init,
    'e_rev_E':     0.,
    'e_rev_I':    -92.,
    'tau_slow':    10.0,
    'tau_syn_E_slow': 100.0,
    'tau_syn_I_slow': 100.0,
    'v_activate_slow': -100.0,
    # 'v_thresh_max': v_thr,
    # 'v_thresh_min': v_thr,
    'v_thresh_max': v_thr+5.0,
    'v_thresh_min': v_thr,
    'thresh_mult_up': 2.0,
    'thresh_mult_down': 0.99975,

}

pynn.setup(timestep=sim_timestep, use_cpu=True)
neuron_class = pynn.IF_curr_exp_slow
neuron_class = pynn.IF_cond_exp_slow

neurons = pynn.Population(N_NEURONS,
            neuron_class(**neuron_parameters),
            label='Target'
          )
neurons.record(['spikes', 'v', 'v_slow', 'dvdt', 'v_thresh'])

pynn.initialize(neurons, v=v_init, v_slow=v_init, v_slow_old=v_init)


inputs = pynn.Population(N_NEURONS,
            pynn.SpikeSourcePoisson(rate=100.0),
            label='Input'
         )
doper = pynn.Population(N_NEURONS,
            pynn.SpikeSourceArray(spike_times=[60.0]),
            label='Feedback'
         )


syn = pynn.StaticSynapse(weight=w*0.01, delay=syn_delay)
proj = pynn.Projection(doper, neurons,
        pynn.OneToOneConnector(), syn,
        receptor_type='excitatory_slow')
wdep = pynn.AdditiveWeightDependence(w_min=0.0, w_max=w)
tdep = pynn.DVDTRule(tau_minus=10.0, tau_plus=10.0,
                          A_plus=1.0, A_minus=1.0)
syn = pynn.DVDTPlasticity(
# syn = pynn.STDPMechanism(
    weight_dependence=wdep,
    timing_dependence=tdep,
    weight=w, delay=syn_delay)

proj = pynn.Projection(inputs, neurons,
        pynn.OneToOneConnector(), syn,
        receptor_type='excitatory')

tsim = 1000.0
pynn.run(tsim)

data = neurons.get_data()
if len(data.segments):
    data = data.segments[0]
    out_spikes = np.array(data.spiketrains)

    pynn.end()

    plt.figure()
    ax = plt.subplot(1,1,1)
    plt.suptitle('Spikes')
    plot_spiketrains(data)
    plt.xlabel("time (%s)" % data.analogsignals[0].times.units._dimensionality.string)
    ax.set_xlim(0, tsim)

    # plt.figure()
    for arr in data.analogsignals:
        plt.figure()
        plt.suptitle('%s'%arr.name)
        for i in range(arr.shape[1]):

            plot_signal(arr, i)

        plt.grid()
        plt.xlabel("time (%s)" % arr.times.units._dimensionality.string)


    plt.show()




