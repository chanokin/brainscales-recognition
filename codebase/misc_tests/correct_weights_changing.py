from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import matplotlib.pyplot as plt
import pynn_genn as sim
import sys

backend = 'genn'
neuron_class = sim.IF_cond_exp
e_rev = 92 #mV
base_params = {
    'cm': 0.09,  # nF
    'v_reset': -90.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'e_rev_I': -e_rev, #mV
    'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms
}

n_pre = 10
n_post = 5

pre_times = [[] for _ in range(n_pre)]
pre_times[2].append(11)
pre_times[4].append(11)
pre_times[6].append(10)
pre_times[8].append(13)

post_times = [[] for _ in range(n_post)]
post_times[1].append(11) #12 - 1
post_times[3].append(14) #15 - 1


run_time = 50
timestep = 1.0
ms_to_ts = 1./timestep
ts_to_ms = 1./ms_to_ts
spike_t = 10

sim.setup(timestep=timestep, min_delay=timestep,
                 extra_params={'use_cpu': True})


pre = sim.Population(n_pre, sim.SpikeSourceArray(spike_times=pre_times))
post = sim.Population(n_post, neuron_class(**base_params))
post_stim = sim.Population(n_post, sim.SpikeSourceArray(spike_times=post_times))

post.record('spikes')
post.record('v')

t_plus = 10.0
t_minus = 10.0
a_plus = 0.5
a_minus = 1.0
w_max = 0.001
stdp = {
    'timing_dependence': {
        'tau_plus': t_plus,
        'tau_minus': t_minus,
        # 'tau_minus': 33.7,
        'A_plus': a_plus, 'A_minus': a_minus,
    },
    'weight_dependence': {
            # 'w_min': (static_w['KC to DN'])/10.0,
            'w_min': 0.0,
            'w_max': w_max,
            # 'w_max': (static_w['KC to DN']),
    },
}

synapse = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(**stdp['timing_dependence']),
    weight_dependence=sim.AdditiveWeightDependence(**stdp['weight_dependence']),
    weight=None, delay=None)

w = [[i, j, w_max/2.0, 1] \
                for i in range(n_pre) for j in range(n_post)]
w_start = np.zeros((n_pre, n_post))
for i, j, v, d in w:
    w_start[i, j] = v
w_start = w_start.flatten()
proj = sim.Projection(pre, post, sim.FromListConnector(conn_list=w),
                      synapse_type=synapse, receptor_type='excitatory')

sim.Projection(post_stim, post, sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=0.1, delay=1.0),
               receptor_type='excitatory')

n_divs = 2
div_run_time = np.ceil(run_time / float(n_divs))
print("run time per div ", div_run_time)
weights = [w_start]
spikes = []
volts = []
print("")
print("")
t = 0.0
for i in range(n_divs):

    t = sim.run(div_run_time)
    sys.stdout.write('loop {} - t {} - run time {}\n'.format(i, t, div_run_time))
    sys.stdout.flush()

    weights.append(proj.getWeights(format='array').flatten())

weights = np.array(weights)
# print(weights)
# print(post.get_data())

spiketrains = post.get_data().segments[0].spiketrains
spikes = [[] for _ in spiketrains]
for train in spiketrains:
    spikes[int(train.annotations['source_index'])][:] = \
        [float(t) for t in train]


volts = post.get_data().segments[0].filter(name='v')[0]
sim.end()

start_t = 0
end_t = run_time


epsilon = 0.0001
wdiff = weights[1, :] - weights[0, :]
changed = np.where(np.abs(wdiff) > epsilon)[0]
w_pre = changed // n_post
w_post = changed % n_post

print("num pre", n_pre)
print("num post", n_post)

print("changed weight id (flattened)", changed)
print("w pre", w_pre)
print("w post ", w_post)


pre_ids = []
for i, times in enumerate(pre_times):
    if len(times):
        pre_ids.append(i)

post_ids = []
for i, times in enumerate(post_times):
    if len(times):
        post_ids.append(i)


print("pre spiked", pre_ids)
print("post spiked", post_ids)

w_should = [i*n_post + j for i in pre_ids for j in post_ids]
w_should_did = np.intersect1d(changed, w_should)
print("pre which changed and should have changed", w_should_did//n_post)
print("post which changed and should have changed", w_should_did%n_post)

if len(changed) > len(w_should):
    w_shouldnt_did = np.setdiff1d(changed, w_should)
else:
    w_shouldnt_did = np.setdiff1d(w_should, changed)

print("pre which changed and should NOT have changed", w_shouldnt_did//n_post)
print("post which changed and should NOT have changed", w_shouldnt_did%n_post)


plt.figure(figsize=(20, 7))
ax = plt.subplot(1, 2, 1)
plt.plot(weights)
xticks = np.array( ax.get_xticks() )
ax.set_xticklabels(xticks * div_run_time)
ax.set_xlabel('time [ms]')
ax.set_ylabel('weights')

ax.grid()

ax = plt.subplot(1, 2, 2)
for i, v in enumerate(volts.T):
    #v = v[start_t:end_t]
    plt.plot( start_t + np.arange(len(v)), v, label='w({}): {}'.format(i, w[i][2]) )

for times in post_times:
    for t in times:
        plt.axvline(1+t*ms_to_ts, linestyle='--', color='blue', linewidth=1.0)

for times in pre_times:
    for t in times:
        plt.axvline(t*ms_to_ts, linestyle='-.', color='green', linewidth=1.0)

plt.axhline(base_params['v_thresh'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_rest'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_reset'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['e_rev_I'], linestyle='--', color='cyan', linewidth=1.0)

ax.grid()
plt.legend()


plt.show()

