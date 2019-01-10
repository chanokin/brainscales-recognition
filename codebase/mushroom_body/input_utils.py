from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict
import numpy as np

def generate_input_vectors(num_vectors, dimension, on_probability, seed=1):
    np.random.seed(seed)
    vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    np.random.seed()
    return vecs

def generate_samples(input_vectors, num_samples, prob_noise, seed=1):
    np.random.seed(seed)
    
    samples = None
    for i in range(input_vectors.shape[0]):
        samp = np.tile(input_vectors[i, :], (num_samples, 1)).astype('int')
        dice = np.random.uniform(0., 1., samp.shape)
        whr = np.where(dice < prob_noise)
        samp[whr] = 1 - samp[whr]
        if samples is None:
            samples = samp
        else:
            samples = np.append(samples, samp, axis=0)

    np.random.seed()
    return samples

def samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, seed=1,
    randomize_samples=False):
    np.random.seed(seed)
    t = 0
    spike_times = [[] for _ in range(samples.shape[-1])]
    if randomize_samples:
        rand_idx = np.random.choice(np.arange(samples.shape[0]), size=samples.shape[0],
                    replace=False)
    else:
        rand_idx = np.arange(samples.shape[0])

    for idx in rand_idx:
        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        ts = t + start_dt + np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size) 
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt
    np.random.seed()
    return spike_times
