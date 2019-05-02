import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pynn_genn as pynn
from pyNN.utility.plotting import *


def assemble_dvdt(pre_spikes, post_spikes, post_dvdt, max_dt, timestep, decimals):
    dtdv = {np.around(dt, decimals=decimals): [] \
            for dt in np.arange(-max_dt, max_dt + timestep, timestep)}
    n_neurons = len(pre_spikes)
    for nid in range(n_neurons):
        posts = post_spikes[nid]
        if len(posts) > 1:
            continue

        pres = pre_spikes[nid]
        dvdt = post_dvdt[nid]

        for post_t in posts:
            post_t = float(post_t)
            whr = np.where(np.logical_and(
                (post_t - max_dt) <= pres, pres < (post_t + max_dt)))
            if len(whr[0]):
                pre_ts = np.array([float(ttt) for ttt in pres[whr]])
                pre_loc = (pre_ts / timestep).astype('int')
                dvs = dvdt[pre_loc]
                dts = np.around(pre_ts - post_t, decimals=decimals)

                for i in range(len(dvs)):
                    dtdv[dts[i]].append(dvs[i])

    return dtdv


def average_dtdv(dtdv):
    avg = {}
    for dt in dtdv:
        avg[dt] = np.mean(dtdv[dt])
    return avg


def plot_spiketrains(segment, label=None):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_index']
        plt.plot(spiketrain, y, '.', label=label)
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=True)


def plot_signal(signal, index, colour=None):
    # label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], color=colour)#, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))


N_NEURONS = 10000
test = N_NEURONS <= 10
start_w = 0.001
noise_w = 0.01
max_w = 0.2
noise_rate = 100.0
input_rate = 10.0
sim_timestep = 0.1
sim_run_time = 200.
syn_delay = sim_timestep
v_init = -50.0

neuron_parameters = {
    'v_thresh':    v_init + 5.0,
    'tau_m':       10.,
    'tau_refrac':  1.0,
    'v_reset':     v_init - 5.0,  #hdbrgs
    'tau_syn_E':   1.0,
    'tau_syn_I':   1.0,
    'i_offset':    0.0,
    #ESS - BrainScaleS
    'cm':          0.2,
    'v_rest':     v_init,
    'e_rev_E':     0.,
    'e_rev_I':    -92.,
    'tau_slow':    20.0,
    'tau_syn_E_slow': 100.0,
    'tau_syn_I_slow': 100.0,
    'v_activate_slow': -100.0,

}

pynn.setup(timestep=sim_timestep, use_cpu=True)

# neuron_class = pynn.IF_curr_exp_slow
neuron_class = pynn.IF_cond_exp_slow

neurons = pynn.Population(N_NEURONS,
            neuron_class(**neuron_parameters),
            label='Target'
          )
neurons.record(['spikes', 'v', 'v_slow', 'dvdt'])

pynn.initialize(neurons, v=v_init, v_slow=v_init, v_slow_old=v_init)



inputs = pynn.Population(N_NEURONS,
            pynn.SpikeSourcePoisson(rate=100.0),
            label='Input'
         )
inputs.record('spikes')
# doper = pynn.Population(N_NEURONS,
#             pynn.SpikeSourceArray(spike_times=[60.0]),
#             label='Feedback'
#          )

# proj = pynn.Projection(doper, neurons,
#         pynn.OneToOneConnector(), syn,
#         receptor_type='excitatory_slow')

noise_pop = pynn.Population(N_NEURONS,
            pynn.SpikeSourcePoisson(rate=noise_rate),
            label='Noise')
noise_pop.record('spikes')

syn = pynn.StaticSynapse(weight=noise_w, delay=syn_delay)
proj = pynn.Projection(noise_pop, neurons,
                       pynn.OneToOneConnector(),
                       syn, receptor_type='excitatory')


inh_noise_pop = pynn.Population(N_NEURONS,
            pynn.SpikeSourcePoisson(rate=noise_rate),
            label='Noise')
inh_noise_pop.record('spikes')

syn = pynn.StaticSynapse(weight=noise_w * 0.75, delay=syn_delay)
proj = pynn.Projection(inh_noise_pop, neurons,
                       pynn.OneToOneConnector(),
                       syn, receptor_type='inhibitory')


wdep = pynn.AdditiveWeightDependence(w_min=0.0, w_max=max_w)
tdep = pynn.DVDTRule(tau_minus=10.0, tau_plus=10.0,
                     A_plus=1.0, A_minus=1.0)
syn = pynn.DVDTPlasticity(
    weight_dependence=wdep,
    timing_dependence=tdep,
    weight=start_w, delay=syn_delay)

proj = pynn.Projection(inputs, neurons,
        pynn.OneToOneConnector(), syn,
        receptor_type='excitatory')

pynn.run(sim_run_time)

post_data = neurons.get_data()
noise_data = noise_pop.get_data()
input_data = inputs.get_data()

pynn.end()




if test and len(post_data.segments):
    print([len(spks) for spks in post_data.segments[0].spiketrains])
    data = post_data.segments[0]
    out_spikes = np.array(data.spiketrains)


    plt.figure()
    plt.suptitle('Spikes')
    plot_spiketrains(data)
    plt.xlabel("time (%s)" % data.analogsignals[0].times.units._dimensionality.string)

    # plt.figure()
    for arr in data.analogsignals:
        if arr.name is not 'v':
            continue
        plt.figure()
        plt.suptitle('%s' % arr.name)
        for i in range(arr.shape[1]):
            plot_signal(arr, i)

        plt.grid()
        plt.xlabel("time (%s)" % arr.times.units._dimensionality.string)

    plt.show()





else:
    np.savez_compressed('plasticity_curve_data.npz',
                        post_data=post_data,
                        noise_data=noise_pop,
                        input_data=input_data,
                        start_w=start_w,
                        n_neurons=N_NEURONS,
                        neuron_parameters=neuron_parameters,
                        timestep=sim_timestep,
                        runtime=sim_run_time,
                        )

    timestep = sim_timestep
    runtime = sim_run_time
    size_time = int(runtime / timestep)
    max_dt = 40.0

    post = post_data
    pre = input_data

    pre_spikes = pre.segments[0].spiketrains
    post_spikes = post.segments[0].spiketrains
    for arr in post.segments[0].analogsignals:
        if arr.name == 'dvdt':
            post_dvdt = arr.T

    dtdv = assemble_dvdt(pre_spikes, post_spikes, post_dvdt, max_dt, timestep, 1)
    avg_dtdv = average_dtdv(dtdv)

    vals = np.array(avg_dtdv.values())
    vals[np.isnan(vals)] = 0.0
    max_dv = np.max(np.abs(vals))
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    for dt in avg_dtdv:
        if np.isnan(avg_dtdv[dt]):
            print(dt)
            continue
        plt.plot(dt, avg_dtdv[dt], '.b', alpha=0.2)

    ax.set_ylim(-max_dv * 1.1, max_dv * 1.1)
    ax.set_xlabel('$\Delta t [t_{post} - t_{pre}, ms]$', fontsize=16)
    ax.set_ylabel('$\Delta w$', fontsize=16)

    # dts = sorted(avg_dtdv.keys())
    # avgs = []
    # for dt in dts:
    #     avgs.append(avg_dtdv[dt])
    # plt.plot(dts, avgs, 'r')
    plt.tight_layout()
    plt.savefig('average_weight_change_centered_at_post_t.pdf')
    plt.show()