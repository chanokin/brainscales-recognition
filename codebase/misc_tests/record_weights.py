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
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'e_rev_I': -e_rev, #mV
    'e_rev_E': 0.,#e_rev, #mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms
}


n_neurons = 3
run_time = 500
timestep = 1.0
spike_t = 10

sim.setup(timestep=timestep, min_delay=timestep,
                 extra_params={'use_cpu': True})

spike_t = 10
spike_times = [[spike_t] for _ in range(n_neurons)]
pre = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=spike_times))
post = sim.Population(n_neurons, neuron_class(**base_params))

post.record('spikes')
post.record('v')


w = [[i, j, (i*n_neurons + j) * 0.001, 1] for i in range(n_neurons) for j in range(n_neurons)]
proj = sim.Projection(pre, post, sim.FromListConnector(conn_list=w), receptor_type='excitatory')

n_divs = 4
div_run_time = np.ceil(run_time / float(n_divs))
print("run time per div ", div_run_time)
weights = []
spikes = []
volts = []
print("")
print("")
t = 0.0
for i in range(n_divs):

    t = sim.run(div_run_time)
    sys.stdout.write('loop {} - t {} - run time {}\n'.format(i, t, div_run_time))
    sys.stdout.flush()

    data = post.get_data()


print(weights)
print(post.get_data())

spiketrains = post.get_data().segments[0].spiketrains
spikes = [[] for _ in spiketrains]
for train in spiketrains:
    spikes[int(train.annotations['source_index'])][:] = \
        [float(t) for t in train]


volts = post.get_data().segments[0].filter(name='v')[0]
sim.end()

start_t = 0
end_t = run_time
plt.figure(figsize=(20, 7))
ax = plt.subplot(1, 2, 1)
for i, times in enumerate(spikes):
    times = np.array(times)
    whr = np.where(np.logical_and(times >= start_t, times < end_t))
    if len(whr[0]):
        print(whr)
        times = times[whr]
        plt.plot(times, i * np.ones_like(times), '.b')

# ax.set_xlim(-1, run_time + 1)
ax.set_ylim(-1, n_neurons + 1)
ax.grid()

ax = plt.subplot(1, 2, 2)
for i, v in enumerate(volts.T):
    v = v[start_t:end_t]
    plt.plot( start_t + np.arange(len(v)), v, label='w({}): {}'.format(i, w[i][2]) )


plt.axvline(spike_t, linestyle='-.', color='green', linewidth=2.0)
plt.axhline(base_params['v_thresh'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_rest'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_reset'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['e_rev_I'], linestyle='--', color='cyan', linewidth=1.0)

ax.grid()
plt.legend()

plt.show()

