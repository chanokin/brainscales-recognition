from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import matplotlib.pyplot as plt
# import pynn_genn as sim
from spikevo.pynn_transforms import PyNNAL



backend = 'genn'
neuron_class = 'IF_curr_exp'
neuron_class = 'IF_cond_exp'

base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms
}


n_neurons = 49
run_time = 50
timestep = 0.1

pynnx = PyNNAL(backend)
pynnx.setup(timestep=timestep, min_delay=timestep,
            per_sim_params={'use_cpu': True})


spike_ts = np.unique(np.random.randint(10, 20, size=(1, 10)))
spike_times = [spike_ts if np.random.uniform() <= 1.0 else [] for _ in range(n_neurons)]
pre = pynnx.Pop(n_neurons, 'SpikeSourceArray', {'spike_times': spike_times})
post = pynnx.Pop(n_neurons, neuron_class, base_params)
pynnx.set_recording(post, 'spikes')
pynnx.set_recording(post, 'v')

inh_spike_times = [[np.random.randint(13, 15)] if np.random.uniform() <= 0.3 else [] \
                   for _ in range(n_neurons)]
inh_pre = pynnx.Pop(n_neurons, 'SpikeSourceArray', {'spike_times': inh_spike_times})
inh_post = pynnx.Pop(n_neurons, neuron_class, base_params)
inh_post.record('spikes')

w = 0.05
proj = pynnx.Proj(pre, post, 'OneToOneConnector',
                  weights=w, target='excitatory')

proj = pynnx.Proj(inh_pre, inh_post, 'OneToOneConnector',
                  weights=w, target='excitatory')

proj = pynnx.Proj(inh_post, post, 'OneToOneConnector',
                  weights=10.*w, target='inhibitory')

pynnx.run(run_time)

spikes = pynnx.get_spikes(post)

inh_spikes = pynnx.get_spikes(inh_post)

volts = np.array(pynnx.get_record(post, 'v'))

pynnx.end()

start_t = 0
end_t = start_t + run_time
plt.figure(figsize=(20, 7))
ax = plt.subplot(1, 2, 1)
for i, times in enumerate(spike_times):
    times = np.array(times)
    whr = np.where(np.logical_and(times >= start_t, times < end_t))
    if len(whr[0]):
        # print(whr)
        times = times[whr]
        plt.plot(times, i * np.ones_like(times), '.g', markersize=2)

for i, times in enumerate(spikes):
    times = np.array(times)
    whr = np.where(np.logical_and(times >= start_t, times < end_t))
    if len(whr[0]):
        # print(whr)
        times = times[whr]
        plt.plot(times, i * np.ones_like(times), '.b', markersize=2)

for i, times in enumerate(inh_spike_times):
    times = np.array(times)
    whr = np.where(np.logical_and(times >= start_t, times < end_t))
    if len(whr[0]):
        # print(whr)
        times = times[whr]+2
        plt.plot(times, i * np.ones_like(times), '_r', markersize=10, markeredgewidth=1.)


ax.set_xlim(-1, run_time + 1)
ax.set_ylim(-1, n_neurons + 1)
ax.grid()

time_to_idx = 1./timestep
ax = plt.subplot(1, 2, 2)
for i, v in enumerate(volts.T):
    start_idx, end_idx = int(start_t * time_to_idx), int(end_t * time_to_idx)
    v = v[start_idx:end_idx]
    plt.plot( start_t + np.arange(len(v)), v,
              label='v({})'.format(i)
              )

# plt.axvline(spike_t, linestyle='-.', color='green', linewidth=2.0)
plt.axhline(base_params['v_thresh'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_rest'], linestyle='--', color='cyan', linewidth=1.0)
plt.axhline(base_params['v_reset'], linestyle='--', color='cyan', linewidth=1.0)

ax.grid()
plt.legend()

plt.show()

