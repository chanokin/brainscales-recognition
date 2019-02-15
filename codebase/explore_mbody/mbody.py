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
from utils import *

def gain_control_list(input_size, horn_size, max_w, cutoff=0.75):
    n_cutoff = 15#int(cutoff*horn_size)
    matrix = np.ones((input_size*horn_size, 4))
    matrix[:, 0] = np.repeat(np.arange(input_size), horn_size)
    matrix[:, 1] = np.tile(np.arange(horn_size), input_size)

    matrix[:, 2] = np.tile( max_w / (n_cutoff + 1.0 + np.arange(horn_size)), input_size)

    return matrix

def output_connection_list(kenyon_size, decision_size, prob_active,
                           active_weight, inactive_scaling, seed=1):
    matrix = np.ones((kenyon_size * decision_size, 4))
    matrix[:, 0] = np.repeat(np.arange(kenyon_size), decision_size)
    matrix[:, 1] = np.tile(np.arange(decision_size), kenyon_size)

    np.random.seed(seed)

    inactive_weight = active_weight * inactive_scaling
    matrix[:, 2] = np.random.normal(inactive_weight, inactive_weight * 0.2,
                                    size=(kenyon_size * decision_size))

    dice = np.random.uniform(0., 1., size=(kenyon_size*decision_size))
    active = np.where(dice <= prob_active)
    matrix[active, 2] = np.random.normal(active_weight, active_weight * 0.2,
                                         size=active[0].shape)

    np.random.seed()

    return matrix

def args_to_str(arguments):
    d = vars(arguments)
    arglist = []
    for arg in d:
        v = str(d[arg])
        if arg.startswith('render'):
            continue
        v = v.replace('.', 'p')
        arglist.append('{}={}'.format(arg, v))

    return '_'.join(arglist)

args = get_args()
pprint(args)

backend = args.backend

# heidelberg's brainscales seems to like these params
e_rev = 92 #mV
base_params = {
    'cm': 0.09, #nF
    'v_reset': -70., #mV
    'v_rest': -65., #mV
    'v_thresh': -55., #mV
    'e_rev_I': -e_rev, #mV
    'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10., #ms
    'tau_refrac': 1., #ms
}

kenyon_parameters = base_params.copy()
kenyon_parameters['tau_syn_E'] = 1.0#ms
kenyon_parameters['tau_syn_I'] = 5.0#ms

horn_parameters = base_params.copy()
horn_parameters['tau_syn_E'] = 1.0#ms

decision_parameters = base_params.copy()
decision_parameters['tau_syn_E'] = 1.0 #ms
# decision_parameters['tau_syn_I'] = 2.5 #ms
decision_parameters['tau_syn_I'] = 5.0 #ms

neuron_params = {
    'base': base_params, 'kenyon': kenyon_parameters,
    'horn': horn_parameters, 'decision': decision_parameters,
}

W2S = args.w2s

sys.stdout.write('Creating input patterns\n')
sys.stdout.flush()


input_vecs = generate_input_vectors(args.nPatternsAL, args.nAL, args.probAL, seed=123)
# input_vecs = generate_input_vectors(10, 100, 0.1)
# pprint(input_vecs)

samples = generate_samples(input_vecs, args.nSamplesAL, args.probNoiseSamplesAL, seed=234,
                           method='exact')
# pprint(samples)

sample_dt, start_dt, max_rand_dt = 50, 25, 1
sample_indices, spike_times = samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt,
                                randomize_samples=args.randomizeSamplesAL, seed=345)

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
pynnx.setup(per_sim_params={'use_cpu': True})

sys.stdout.write('Done!\tCreating simulator abstraction\n\n')
sys.stdout.flush()


sys.stdout.write('Creating populations\n')
sys.stdout.flush()

populations = {
    'antenna': pynnx.Pop(args.nAL, 'SpikeSourceArray',
                         {'spike_times': spike_times}, label='Antennae Lobe'),
    'kenyon': pynnx.Pop(args.nKC, 'IF_cond_exp',
                        kenyon_parameters, label='Kenyon Cell'),
    'horn': pynnx.Pop(args.nLH, 'IF_cond_exp',
                      horn_parameters, label='Lateral Horn'),
    'decision': pynnx.Pop(args.nDN, 'IF_cond_exp',
                          decision_parameters, label='Decision Neurons'),
}

pynnx.set_recording(populations['decision'], 'spikes')
pynnx.set_recording(populations['kenyon'], 'spikes')
pynnx.set_recording(populations['horn'], 'spikes')

sys.stdout.write('Creating projections\n')
sys.stdout.flush()

static_w = {
    'AL to KC': W2S*(100.0/float(args.nAL)),
    # 'AL to LH': W2S*(1./(args.nAL*args.probAL)),
    # 'AL to LH': W2S*(0.787 * (100.0/float(args.nAL))),
    'AL to LH': W2S*(6.5 * (100.0/float(args.nAL))),
    'LH to KC': W2S*(1.925 * (20.0/float(args.nLH))),

    'KC to KC': W2S*(0.1*(2500.0/float(args.nKC))),

    'KC to DN': W2S*(1.5 * (2500.0/float(args.nKC))),
    # 'DN to DN': W2S*(0.4 * (100.0/float(args.nDN))),
    'DN to DN': W2S*(1. * (100.0/float(args.nDN))),
    # 'DN to DN': W2S*(1./1.),
    # 'DN to DN': W2S*(1./(args.nDN)),

}

rand_w = {
    'AL to KC': pynnx.sim.RandomDistribution('normal',
                    (static_w['AL to KC'], static_w['AL to KC']*0.2),
                    pynnx.sim.NumpyRNG(seed=1)),
}

gain_list = gain_control_list(args.nAL, args.nLH, static_w['AL to LH'])

out_list = output_connection_list(args.nKC, args.nDN, args.probKC2DN,
                                  static_w['KC to DN'], args.inactiveScale,
                                  seed=None)

stdp = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': 16.8, 'tau_minus': 33.7},
    },
    'weight_dependence': {
        'name':'MultiplicativeWeightDependence',
        'params': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': 0.0,
            'w_max': (static_w['KC to DN']),
            'A_plus': 0.00001, 'A_minus': 0.00005
        },
    }
}

projections = {
    'AL to KC': pynnx.Proj(populations['antenna'], populations['kenyon'],
                           'FixedProbabilityConnector', weights=rand_w['AL to KC'], delays=1.0,
                           conn_params={'p_connect': args.probAL2KC}, label='AL to KC'),

    # 'AL to LH': pynnx.Proj(populations['antenna'], populations['horn'],
    #                        'FixedProbabilityConnector', weights=static_w['AL to LH'], delays=1.0,
    #                        conn_params={'p_connect': args.probAL2LH}, label='AL to LH'),

    'AL to LH': pynnx.Proj(populations['antenna'], populations['horn'],
                           'FromListConnector', weights=None, delays=None,
                           conn_params={'conn_list': gain_list}, label='AL to LH'),

    'LH to KC': pynnx.Proj(populations['horn'], populations['kenyon'],
                           'AllToAllConnector', weights=static_w['LH to KC'], delays=1.0,
                           conn_params={}, target='inhibitory', label='LH to KC'),

    ### Inhibitory feedback --- kenyon cells
    'KC to KC': pynnx.Proj(populations['kenyon'], populations['kenyon'],
                            'AllToAllConnector', weights=static_w['KC to KC'], delays=1.0,
                            conn_params={}, target='inhibitory', label='KC to KC'),

    'KC to DN': pynnx.Proj(populations['kenyon'], populations['decision'],
                           'FromListConnector', weights=None, delays=None,
                           conn_params={'conn_list': out_list}, label='KC to DN',
                           stdp=stdp),
    ### Inhibitory feedback --- decision neurons
    'DN to DN': pynnx.Proj(populations['decision'], populations['decision'],
                           'AllToAllConnector', weights=static_w['DN to DN'], delays=1.0,
                           conn_params={}, target='inhibitory', label='DN to DN'),
}

sys.stdout.write('Running simulation\n')
sys.stdout.flush()

sim_time = sample_dt * args.nSamplesAL * args.nPatternsAL

pynnx.run(sim_time)


sys.stdout.write('Done!\tRunning simulation\n\n')
sys.stdout.flush()

sys.stdout.write('Getting spikes:\n')
sys.stdout.flush()

sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
k_spikes = pynnx.get_record(populations['kenyon'], 'spikes')

sys.stdout.write('\tDecision\n')
sys.stdout.flush()
out_spikes = pynnx.get_record(populations['decision'], 'spikes')

sys.stdout.write('\tHorn\n')
sys.stdout.flush()
horn_spikes = pynnx.get_record(populations['horn'], 'spikes')

sys.stdout.write('Done!\tGetting spikes\n\n')
sys.stdout.flush()

sys.stdout.write('Getting weights:\n')
sys.stdout.flush()
sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
try:
    final_weights = pynnx.get_weights(projections['KC to DN'])
except:
    final_weights = None
sys.stdout.write('Done!\t Getting weights\n\n')
sys.stdout.flush()


pynnx.end()


sys.stdout.write('Saving experiment\n')
sys.stdout.flush()
# fname = 'mbody-'+args_to_str(args)+'.npz'
fname = 'mbody-experiment.npz'
np.savez_compressed(fname, args=args, sim_time=sim_time,
    input_spikes=spike_times, input_vectors=input_vecs,
    input_samples=samples, sample_indices=sample_indices,
    output_start_connections=out_list, output_end_weights=final_weights,
    lateral_horn_connections=gain_list,
    static_weights=static_w, stdp_params=stdp,
    kenyon_spikes=k_spikes, decision_spikes=out_spikes, horn_spikes=horn_spikes,
    neuron_parameters=neuron_params,
    sample_dt=sample_dt, start_dt=start_dt, max_rand_dt=max_rand_dt,
)
sys.stdout.write('Done!\tSaving experiment\n\n')
sys.stdout.flush()



if args.renderSpikes:
    render_spikes(k_spikes, 'Kenyon activity', 'kenyon_activity.pdf')

    render_spikes(out_spikes, 'Output activity', 'output_activity.pdf')


