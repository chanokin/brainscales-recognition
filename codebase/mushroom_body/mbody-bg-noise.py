from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import datetime
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
import argparse
from pprint import pprint
from args_setup import get_args
from input_utils import *


def gain_control_list(input_size, horn_size, max_w, cutoff=0.7):
    if cutoff is not None:
        n_cutoff = int(cutoff * horn_size)
    else:
        n_cutoff = 15
    matrix = np.ones((input_size * horn_size, 4))
    matrix[:, 0] = np.repeat(np.arange(input_size), horn_size)
    matrix[:, 1] = np.tile(np.arange(horn_size), input_size)

    matrix[:, 2] = np.tile(max_w / (n_cutoff + 1.0 + np.arange(horn_size)), input_size)

    return matrix


def output_connection_list(kenyon_size, decision_size, prob_active, active_weight,
                           inactive_scaling, delay=1, seed=1, clip_to=np.inf):
    matrix = np.ones((kenyon_size * decision_size, 4)) * delay
    matrix[:, 0] = np.repeat(np.arange(kenyon_size), decision_size)
    matrix[:, 1] = np.tile(np.arange(decision_size), kenyon_size)

    np.random.seed(seed)

    inactive_weight = active_weight * inactive_scaling
    matrix[:, 2] = np.clip(np.random.normal(inactive_weight, inactive_weight * 0.2,
                                            size=(kenyon_size * decision_size)),
                           0, clip_to)

    dice = np.random.uniform(0., 1., size=(kenyon_size * decision_size))
    active = np.where(dice <= prob_active)[0]
    matrix[active, 2] = np.clip(np.random.normal(active_weight, active_weight * 0.2,
                                                 size=active.shape),
                                0, clip_to)

    np.random.seed()
    # pprint(np.where(matrix[:, 2] < 0.0))
    return matrix


def output_pairing_connection_list(decision_size, neighbour_distance, weight, delay=1):
    conn_list = []
    half_dist = neighbour_distance // 2
    for nid in range(decision_size):
        for ndist in range(-half_dist, half_dist + 1):
            neighbour = nid + ndist
            if neighbour < 0 or ndist == 0 or neighbour >= decision_size:
                continue
            conn_list.append([nid, neighbour, weight, delay])

    return conn_list

def args_to_str(arguments, stringable=['nAL', 'nKC', 'nDN', 'probAL', 'probNoiseSamplesAL', 'nPatternsAL']):

    d = vars(arguments)
    arglist = []
    for arg in d:
        v = str(d[arg])
        if arg not in stringable:
            continue
        v = v.replace('.', 'p')
        arglist.append('{}_{}'.format(arg, v))

    return '__'.join(arglist)


args = get_args()
pprint(args)

backend = args.backend

neuron_class = 'IF_cond_exp'
neuron_class = 'IF_curr_exp'
# heidelberg's brainscales seems to like these params
e_rev = 92  # mV
# e_rev = 500.0 #mV

if neuron_class == 'IF_cond_exp':
    base_params = {
        'cm': 0.09,  # nF
        'v_reset': -90.,  # mV
        'v_rest': -65.,  # mV
        'v_thresh': -55.,  # mV
        # 'e_rev_I': -e_rev, #mV
        # 'e_rev_E': 0.,#e_rev, #mV
        'tau_m': 10.,  # ms
        'tau_refrac': 5.,  # ms
        'tau_syn_E': 2.0,  # ms
        'tau_syn_I': 3.0,  # ms

    }

    base_params['e_rev_I'] = -e_rev
    base_params['e_rev_E'] = 0.0
else:
    base_params = {
        'cm': 0.2,  # nF
        'v_reset': -80.,  # mV
        'v_rest': -65.,  # mV
        'v_thresh': -50.,  # mV
        # 'e_rev_I': -e_rev, #mV
        # 'e_rev_E': 0.,#e_rev, #mV
        'tau_m': 5.,  # ms
        'tau_refrac': 5.,  # ms
        'tau_syn_E': 2.0,  # ms
        'tau_syn_I': 5.0,  # ms
    }

kenyon_parameters = base_params.copy()
kenyon_parameters['tau_refrac'] = 1.0  # ms
kenyon_parameters['tau_syn_E'] = 1.0  # ms
kenyon_parameters['tau_syn_I'] = 1.0  # ms
kenyon_parameters['tau_m'] = 10.0  # ms
# kenyon_parameters['v_thresh'] = np.random.normal(-53.0, 1.0, size=args.nKC)
# kenyon_parameters['v_reset'] = -65.0
# kenyon_parameters['cm'] = 0.1


horn_parameters = base_params.copy()
horn_parameters['tau_m'] = 5.0
horn_parameters['tau_syn_E'] = 1.0  # ms
# horn_parameters['v_thresh'] = [-50.0 + float(i) for i in range(args.nLH)]
# print(horn_parameters['v_thresh'])

decision_parameters = base_params.copy()
decision_parameters['tau_syn_E'] = 1.0  # ms
# decision_parameters['tau_syn_I'] = 2.5 #ms
decision_parameters['tau_syn_I'] = 2.0  # ms
decision_parameters['tau_refrac'] = 15.0
decision_parameters['tau_m'] = 5.0
decision_parameters['v_reset'] = -100.0

np.random.seed(321)
# decision_parameters['v_thresh'] = np.random.normal(-50.0, 5.0, size=args.nDN)
# decision_parameters['v_reset'] = np.random.normal(-100.0, 10.0, size=args.nDN)
np.random.seed()

fb_parameters = base_params.copy()
fb_parameters['tau_syn_E'] = 1.0  # ms
fb_parameters['tau_syn_I'] = 1.0  # ms
fb_parameters['tau_refrac'] = 10.0


exciter_parameters = base_params.copy()
exciter_parameters['tau_refrac'] = 1.0  # ms
exciter_parameters['tau_syn_E'] = 1.0  # ms
exciter_parameters['tau_syn_I'] = 100.0  # ms
exciter_parameters['tau_m'] = 5.0  # ms
exciter_parameters['v_reset'] = -65.0  # ms

neuron_params = {
    'base': base_params, 'kenyon': kenyon_parameters,
    'horn': horn_parameters, 'decision': decision_parameters,
    'feedback': fb_parameters, 'exciter': exciter_parameters,
}

W2S = args.w2s
if neuron_class == 'IF_curr_exp':
    ###___this works for 100-2500-100
    W2S *= 0.6/ 0.0025
    W2S = 0.4
    ###___this works for 100-1000-100
    # W2S *= 0.9/ 0.0025
    ###___this works for 49-500-49
    # W2S *= 0.5/ 0.0025

# sample_dt, start_dt, max_rand_dt = 10, 5, 2
sample_dt, start_dt, max_rand_dt = 50, 5, 5.0
sim_time = sample_dt * args.nSamplesAL * args.nPatternsAL
timestep = 1.0 if bool(0) else 0.1
regenerate = bool(0) or args.regenerateSamples
record_all = args.recordAllOutputs and args.nSamplesAL <= 50
fixed_loops = args.fixedNumLoops
n_explore_samples = min(args.nPatternsAL * 10, np.round(args.nSamplesAL * args.nPatternsAL * 0.01))
n_exciter_samples = min(args.nPatternsAL * 100, np.round(args.nSamplesAL * args.nPatternsAL * 0.1))
n_test_samples = min(1000, np.round(args.nSamplesAL * args.nPatternsAL * 1.0/6.0))
use_poisson_input = bool(0)
high_dt = 3
low_freq, high_freq = 10, 100
n_other_pattern = int(args.probAL * args.nAL * 1.0)
n_noisy_samples = int(args.nSamplesAL * args.nPatternsAL)# - n_test_samples)

sys.stdout.write('Creating input patterns\n')
sys.stdout.flush()

sys.stdout.write('\tGenerating input vectors\n')
sys.stdout.flush()

input_vecs = generate_input_vectors(args.nPatternsAL, args.nAL, args.probAL, seed=123)
# input_vecs = generate_input_vectors(10, 100, 0.1)
# pprint(input_vecs)
sys.stdout.write('\t\tDone with input vectors\n')
sys.stdout.flush()


sys.stdout.write('\tGenerating samples\n')
sys.stdout.flush()

samples = generate_samples(input_vecs, args.nSamplesAL, args.probNoiseSamplesAL, seed=234,
                           # method='random',
                           method='exact',
                           regenerate=regenerate)


noise_samples = add_noise_to_samples(input_vecs, samples, n_other_pattern, n_noisy_samples,
                                     seed=456, regenerate=regenerate)
# pprint(samples)
sys.stdout.write('\t\tdone with samples\n')
sys.stdout.flush()


sys.stdout.write('\tGenerating spike times\n')
sys.stdout.flush()

if use_poisson_input:
    sample_indices, spike_times = generate_spike_times_poisson(input_vecs, noise_samples,
                                    sample_dt, start_dt, high_dt, high_freq, low_freq,
                                    seed=234, randomize_samples=True, regenerate=bool(0))
else:
    sample_indices, spike_times = samples_to_spike_times(noise_samples, sample_dt, start_dt, max_rand_dt, timestep,
                                                         randomize_samples=args.randomizeSamplesAL, seed=345,
                                                         regenerate=regenerate)




tick_spikes = generate_tick_spikes(samples, sample_dt, start_dt, n_test_samples, delay=25)

sys.stdout.write('\t\tdone with spike times\n')
sys.stdout.flush()

sys.stdout.write('Done!\tCreating input patterns\n\n')
sys.stdout.flush()

if args.renderSpikes:
    render_spikes(spike_times, 'Input samples', 'input_samples.pdf', markersize=1)
# plt.show()

### -------------------------------------------------------------- ###
### -------------------------------------------------------------- ###
### -------------------------------------------------------------- ###

sys.stdout.write('Creating simulator abstraction\n')
sys.stdout.flush()

pynnx = PyNNAL(backend)
pynnx._sim.setup(timestep=timestep, min_delay=timestep,
                 extra_params={'use_cpu': True})
# pynnx.setup(timestep=timestep, min_delay=timestep,
#             per_sim_params={'use_cpu': True})

sys.stdout.write('Done!\tCreating simulator abstraction\n\n')
sys.stdout.flush()

sys.stdout.write('Creating populations\n')
sys.stdout.flush()

populations = {
    'antenna': pynnx.Pop(args.nAL, 'SpikeSourceArray',
                         {'spike_times': spike_times}, label='Antennae Lobe'),
    'kenyon': pynnx.Pop(args.nKC, neuron_class,
                        kenyon_parameters, label='Kenyon Cell'),
    'horn': pynnx.Pop(args.nLH, neuron_class,
                      horn_parameters, label='Lateral Horn'),
    'decision': pynnx.Pop(args.nDN, neuron_class,
                          decision_parameters, label='Decision Neurons'),

    ### add noise to 'explore' classification in the beginning
    'noise': pynnx.Pop(args.nDN, 'SpikeSourcePoisson',
                       {'rate': 100.0, 'start': start_dt, 'duration': n_explore_samples * sample_dt},
                       label='decision noise'),


    ### make neurons spike right before a new pattern is shown
    'tick': pynnx.Pop(1, 'SpikeSourceArray',
                      {'spike_times': tick_spikes}, label='Tick Neurons'),
    'feedback': pynnx.Pop(args.nDN, neuron_class,
                          fb_parameters, label='Feedback Neurons'),

    ### add current if neuron hasn't spiked yet
    'exciter': pynnx.Pop(args.nDN, neuron_class,
                         exciter_parameters, label='Threshold reducer'),

    'exciter src': pynnx.Pop(args.nDN, 'SpikeSourcePoisson',
                       {'rate': 1000.0, 'start': start_dt,
                        'duration': n_exciter_samples * sample_dt},
                       label='threshold reducer source'),

}

pynnx.set_recording(populations['decision'], 'spikes')
np.random.seed()
populations['decision'].initialize(v=np.random.uniform(-120.0, -50.0, size=args.nDN))
pynnx.set_recording(populations['kenyon'], 'spikes')
if record_all:
    pynnx.set_recording(populations['horn'], 'spikes')
    pynnx.set_recording(populations['feedback'], 'spikes')
    pynnx.set_recording(populations['decision'], 'v')
    # pynnx.set_recording(populations['kenyon'], 'v')
    pynnx.set_recording(populations['exciter'], 'spikes')

sys.stdout.write('Creating projections\n')
sys.stdout.flush()

if neuron_class == 'IF_cond_exp':
    static_w = {
        'AL to KC': W2S * 1.5 * (100.0 / float(args.nAL)),
        # 'AL to LH': W2S*(1./(args.nAL*args.probAL)),
        # 'AL to LH': W2S*(0.787 * (100.0/float(args.nAL))),
        'AL to LH': W2S * (10.0 * (100.0 / float(args.nAL))),

        'LH to KC': W2S * (0.25 * (20.0 / float(args.nLH))),

        'KC to KC': W2S * (5.0 * (2500.0 / float(args.nKC))),

        'KC to DN': W2S * (2.0 * (2500.0 / float(args.nKC))),
        # 'KC to DN': W2S*(2.0 * (2500.0/float(args.nKC))),
        # 'DN to DN': W2S*(0.4 * (100.0/float(args.nDN))),
        'DN to DN': W2S * (10.0 * (100.0 / float(args.nDN))),
        # 'DN to DN': W2S*(1./1.),
        # 'DN to DN': W2S*(1./(args.nDN)),
        'NS to DN': W2S * (0.5 * (100.0 / float(args.nDN))),

        'NS to KC': W2S * (5.0 * (2500.0 / float(args.nKC))),

        'DN to FB': W2S * (1.0 * (100.0 / float(args.nDN))),
        'FB to DN': W2S * (1.0 * (100.0 / float(args.nDN))),
        'TK to FB': W2S * (1.0 * (100.0 / float(args.nDN))),

    }
else:
    static_w = {
        'AL to KC': W2S * 1.0 * (100.0/ float(args.nAL)),
        'AL to LH': W2S * (0.001 * (100.0 / float(args.nAL))),
        ### inhibitory
        'LH to KC': -W2S * (0.02 * (20.0 / float(args.nLH))),
        ### inhibitory
        'KC to KC': -W2S * (0.01 * (2500.0 / float(args.nKC))),

        # this works for 100-2500-100
        'KC to DN': W2S * ((0.8 / 1.2) * (2500.0/float(args.nKC))),

        'iKC to DN': -W2S * (1.0 * (2500.0 / float(args.nKC))),
        ### inhibitory
        'DN to DN': -W2S * (10.0 * (100.0 / float(args.nDN))),
        'pair DN to DN': W2S * (0.1 * (100.0 / float(args.nDN))),

        'NS to DN':  W2S * (0.001 * (100.0 / float(args.nDN))),

        'DN to FB': 3.0,
        'FB to DN': 5.0,
        'TK to FB': 3.0,

        'DN to TR': -10.0,
        'TRS to TR': 5.0,
        'TR to DN':  0.5,
    }

rand_w = {
    'AL to KC': static_w['AL to KC'],
    # 'AL to KC': pynnx.sim.RandomDistribution('normal',
    #                                          (static_w['AL to KC'], static_w['AL to KC'] * 0.2),
    #                                          pynnx.sim.NumpyRNG(seed=1)),
}

w_max = (static_w['KC to DN'] * 1.0) * 1.2
# w_max = 0.2
# w_max = W2S*(0.5 * (2500.0/float(args.nKC)))
w_min = -2. * w_max
# w_min = 0.0 * w_max
print("\nw_min = {}\tw_max = {}\n".format(w_min, w_max))

gain_list = gain_control_list(args.nAL, args.nLH, static_w['AL to LH'],
                              # cutoff=None
                              )

out_list = output_connection_list(args.nKC, args.nDN, args.probKC2DN,
                                  static_w['KC to DN'],
                                  # 0.,
                                  args.inactiveScale,
                                  delay=3,
                                  seed=None,
                                  # seed=123456,
                                  # clip_to=np.inf
                                  clip_to=w_max
                                  )

out_neighbours = output_pairing_connection_list(args.nDN, 11, static_w['pair DN to DN'], delay=1)

t_plus = 5.0
t_minus = 15.0
# t_minus = 20.0
a_plus = 0.1
# a_minus = 0.05
a_minus = 0.1


stdp = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': t_plus,
                   'tau_minus': t_minus,
                   # 'tau_minus': 33.7,
                   },
    },
    'weight_dependence': {
        'name': 'AdditiveWeightDependence',
        # 'name':'MultiplicativeWeightDependence',
        'params': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': w_min,
            'w_max': w_max,
            # 'w_max': (static_w['KC to DN']),
            'A_plus': a_plus, 'A_minus': a_minus,
        },
    }
}

stdp_inv = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': t_minus,
                   'tau_minus': t_plus,
                   # 'tau_minus': 33.7,
                   },
    },
    'weight_dependence': {
        'name': 'AdditiveWeightDependence',
        # 'name':'MultiplicativeWeightDependence',
        'params': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': w_min,
            'w_max': w_max,
            # 'w_max': (static_w['KC to DN']),
            'A_plus': -a_minus, 'A_minus': -a_plus,
        },
    }
}

projections = {
    'AL to KC': pynnx.Proj(populations['antenna'], populations['kenyon'],
                           'FixedProbabilityConnector', weights=rand_w['AL to KC'], delays=4.0,
                           conn_params={'p_connect': args.probAL2KC}, label='AL to KC'),

    # 'AL to LH': pynnx.Proj(populations['antenna'], populations['horn'],
    #                        'FixedProbabilityConnector', weights=static_w['AL to LH'], delays=1.0,
    #                        conn_params={'p_connect': args.probAL2LH}, label='AL to LH'),

    'AL to LH': pynnx.Proj(populations['antenna'], populations['horn'],
                           'FromListConnector', weights=None, delays=None,
                           conn_params={'conn_list': gain_list}, label='AL to LH'),

    'LH to KC': pynnx.Proj(populations['horn'], populations['kenyon'],
                           'AllToAllConnector', weights=static_w['LH to KC'], delays=timestep,
                           conn_params={}, target='inhibitory', label='LH to KC'),

    ### Inhibitory feedback --- kenyon cells
    'KC to KC': pynnx.Proj(populations['kenyon'], populations['kenyon'],
                            'AllToAllConnector', weights=static_w['KC to KC'], delays=timestep,
                            conn_params={'allow_self_connections': False}, target='inhibitory', label='KC to KC'),
    # 'KC to KC': pynnx.Proj(populations['kenyon'], populations['kenyon'],
    #                         'FixedProbabilityConnector', weights=static_w['KC to KC'], delays=timestep,
    #                         conn_params={'p_connect': 0.8}, target='inhibitory', label='KC to KC'),

    'KC to DN': pynnx.Proj(populations['kenyon'], populations['decision'],
                           'FromListConnector', weights=None, delays=None,
                           conn_params={'conn_list': out_list}, label='KC to DN',
                           stdp=stdp),

    # 'KC to DN inv': pynnx.Proj(populations['kenyon'], populations['decision'],
    #                        'FromListConnector', weights=None, delays=None,
    #                        conn_params={'conn_list': out_list}, label='KC to DN',
    #                        stdp=stdp_inv),

    # 'iKC to DN': pynnx.Proj(populations['kenyon'], populations['decision'],
    #                        'FixedProbabilityConnector', weights=static_w['iKC to DN'], delays=timestep,
    #                        conn_params={'p_connect': 0.1}, target='inhibitory', label='iKC to DN'),

    ### Inhibitory feedback --- decision neurons
    # 'DN to DN': pynnx.Proj(populations['decision'], populations['decision'],
    #                        'FixedProbabilityConnector', weights=static_w['DN to DN'], delays=timestep,
    #                        conn_params={'p_connect': 0.5}, target='inhibitory', label='DN to DN'),

    'DN to DN': pynnx.Proj(populations['decision'], populations['decision'],
                           'AllToAllConnector', weights=static_w['DN to DN'], delays=timestep,
                           conn_params={'allow_self_connections': True}, target='inhibitory', label='DN to DN'),

    # 'pair DN to DN': pynnx.Proj(populations['decision'], populations['decision'],
    #                        'FromListConnector', weights=None, delays=None,
    #                        conn_params={'conn_list': out_neighbours},
    #                        target='excitatory', label='pair DN to DN'),

    ### Add noise to decision neurons
    # 'NS to DN': pynnx.Proj(populations['noise'], populations['decision'],
    #                        'FixedProbabilityConnector', weights=static_w['NS to DN'], delays=1.0,
    #                        conn_params={'p_connect': 0.05}, target='excitatory',
    #                        label='NS to DN'),

    # 'NS to DN': pynnx.Proj(populations['noise'], populations['decision'],
    #                        'OneToOneConnector', weights=static_w['NS to DN'], delays=1.0,
    #                        target='excitatory', label='NS to DN'),

    ### inhibitory noise
    # 'iNS to DN': pynnx.Proj(populations['noise'], populations['decision'],
    #                        'FixedProbabilityConnector', weights=-static_w['NS to DN'], delays=1.0,
    #                        conn_params={'p_connect': 0.05}, target='inhibitory',
    #                        label='iNS to DN'),

    ### add noise to kenyon cells
    # 'NS to KC': pynnx.Proj(populations['noise'], populations['kenyon'],
    #                        'FixedProbabilityConnector', weights=static_w['NS to KC'], delays=1.0,
    #                        conn_params={'p_connect': 0.1}, target='excitatory',
    #                        label='NS to KC'),

    ### make decision spike just before the next pattern to reduce weights corresponding to that input
    'DN to FB': pynnx.Proj(populations['decision'], populations['feedback'],
                           'OneToOneConnector', weights=static_w['DN to FB'], delays=15.0,
                           target='excitatory', label='DN to FB'),

    'FB to DN': pynnx.Proj(populations['feedback'], populations['decision'],
                           'OneToOneConnector', weights=static_w['FB to DN'], delays=15.0,
                           target='excitatory', label='FB to DN'),

    'TK to FB': pynnx.Proj(populations['tick'], populations['feedback'],
                           'AllToAllConnector', weights=static_w['TK to FB'], delays=1.0,
                           target='excitatory', label='TK to FB'),

    ### have some more current comming into decicions if they have not spiked recently
    'TR to DN': pynnx.Proj(populations['exciter'], populations['decision'],
                           'OneToOneConnector', weights=static_w['TR to DN'], delays=1.0,
                           target='excitatory', label='TR to DN'),

    'DN to TR': pynnx.Proj(populations['decision'], populations['exciter'],
                           'OneToOneConnector', weights=static_w['DN to TR'], delays=timestep,
                           target='inhibitory', label='DN to TR'),

    'TRS to TR': pynnx.Proj(populations['exciter src'], populations['exciter'],
                           'OneToOneConnector', weights=static_w['TRS to TR'], delays=1.0,
                           target='excitatory', label='TR to TR'),

}

sys.stdout.write('Running simulation\n')
sys.stdout.flush()

starting_weights = np.zeros((args.nKC, args.nDN))
for i, j, v, d in out_list:
    starting_weights[int(i), int(j)] = v
starting_weights = starting_weights.flatten()
weights = [starting_weights]

if fixed_loops == 0:
    # weight_sample_dt = 10.
    weight_sample_dt = float(sample_dt * (args.nSamplesAL * 0.1))
    n_loops = np.ceil(sim_time / weight_sample_dt)
else:
    n_loops = fixed_loops
    weight_sample_dt = np.ceil(sim_time / float(n_loops))

print("num loops = {}\ttime per loop {}".format(n_loops, weight_sample_dt))
now = datetime.datetime.now()
sys.stdout.write("\tstarting time is {:02d}:{:02d}:{:02d}\n".format(now.hour, now.minute, now.second))
sys.stdout.flush()

t0 = time.time()
for loop in np.arange(n_loops):
    sys.stdout.write("\trunning loop {} of {}\t".format(loop + 1, n_loops))
    sys.stdout.flush()

    loop_t0 = time.time()
    now = datetime.datetime.now()
    sys.stdout.write("starting {:02d}:{:02d}:{:02d}\t".format(now.hour, now.minute, now.second))
    sys.stdout.flush()


    pynnx.run(weight_sample_dt)


    secs_to_run = time.time() - loop_t0
    mins_to_run = secs_to_run // 60
    secs_to_run -= mins_to_run * 60
    hours_to_run = mins_to_run // 60
    mins_to_run -= hours_to_run * 60
    secs_to_run, mins_to_run, hours_to_run = int(secs_to_run), int(mins_to_run), int(hours_to_run)

    sys.stdout.write('lasted {:02d}h: {:02d}m: {:02d}s\n'.format(hours_to_run, mins_to_run, secs_to_run))
    sys.stdout.flush()

    tmp_w = pynnx.get_weights(projections['KC to DN'])
    # print(loop, tmp_w.shape)
    weights.append(tmp_w.flatten())

post_horn = []#pynnx.get_weights(projections['AL to LH'])
secs_to_run = time.time() - t0

mins_to_run = secs_to_run // 60
secs_to_run -= mins_to_run * 60
hours_to_run = mins_to_run // 60
mins_to_run -= hours_to_run * 60
secs_to_run, mins_to_run, hours_to_run = int(secs_to_run), int(mins_to_run), int(hours_to_run)

sys.stdout.write('\n\nDone!\tRunning simulation - lasted {:02d}h: {:02d}m: {:02d}s\n\n'. \
                 format(hours_to_run, mins_to_run, secs_to_run))
sys.stdout.flush()

sys.stdout.write('Getting spikes:\n')
sys.stdout.flush()

sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
k_spikes = pynnx.get_record(populations['kenyon'], 'spikes')

sys.stdout.write('\tDecision\n')
sys.stdout.flush()
out_spikes = pynnx.get_record(populations['decision'], 'spikes')

if record_all:
    sys.stdout.write('\tHorn\n')
    sys.stdout.flush()
    horn_spikes = pynnx.get_record(populations['horn'], 'spikes')

    sys.stdout.write('\tFeedback\n')
    sys.stdout.flush()
    fb_spikes = pynnx.get_record(populations['feedback'], 'spikes')

    sys.stdout.write('\tExciter\n')
    sys.stdout.flush()
    exciter_spikes = pynnx.get_record(populations['exciter'], 'spikes')

else:
    horn_spikes = [[]]
    fb_spikes = [[]]
    exciter_spikes = [[]]

sys.stdout.write('Done!\tGetting spikes\n\n')
sys.stdout.flush()


if record_all:
    sys.stdout.write('Getting voltages\n')
    sys.stdout.flush()

    dn_voltage = pynnx.get_record(populations['decision'], 'v')
    # kc_voltage = pynnx.get_record(populations['kenyon'], 'v')
    kc_voltage = [np.array([[0, 0]])]

    sys.stdout.write('Done!\tGetting voltages\n\n')
    sys.stdout.flush()

else:
    dn_voltage = [np.array([[0, 0]])]
    kc_voltage = [np.array([[0, 0]])]

sys.stdout.write('Getting weights:\n')
sys.stdout.flush()
sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
# try:
final_weights = weights[-1]
if 'KC to DN inv' in projections:
    final_weights_inv = pynnx.get_weights(projections['KC to DN inv'])
else:
    final_weights_inv = None

sys.stdout.write('Done!\t Getting weights\n\n')
sys.stdout.flush()

pynnx.end()

sys.stdout.write('Saving experiment\n')
sys.stdout.flush()
# fname = 'mbody-'+args_to_str(args)+'.npz'
fname = 'bgn-mbody-experiment.npz'
np.savez_compressed(fname, args=args, sim_time=sim_time,
                    input_spikes=spike_times, input_vectors=input_vecs,
                    input_samples=samples, sample_indices=sample_indices,
                    input_samples_noisy=noise_samples,
                    output_start_connections=out_list, lateral_horn_connections=gain_list,
                    output_end_weights=final_weights, output_end_weights_inv=final_weights_inv,
                    static_weights=static_w, stdp_params=stdp,
                    kenyon_spikes=k_spikes, decision_spikes=out_spikes, horn_spikes=horn_spikes,
                    neuron_parameters=neuron_params,
                    sample_dt=sample_dt, start_dt=start_dt, max_rand_dt=max_rand_dt,
                    dn_voltage=dn_voltage, kc_voltage=kc_voltage,
                    high_dt=high_dt,
                    low_freq=low_freq, high_freq=high_freq,
                    weights=weights, weight_sample_dt=weight_sample_dt,
                    timestep=timestep,
                    post_horn_weights=post_horn,
                    tick_spikes=tick_spikes,
                    fb_spikes=fb_spikes,
                    exciter_spikes=exciter_spikes,
                    n_test_samples=n_test_samples,
                    )
sys.stdout.write('Done!\tSaving experiment\n\n')
sys.stdout.flush()

if args.renderSpikes:
    render_spikes(k_spikes, 'Kenyon activity', 'kenyon_activity.pdf')

    render_spikes(out_spikes, 'Output activity', 'output_activity.pdf')
