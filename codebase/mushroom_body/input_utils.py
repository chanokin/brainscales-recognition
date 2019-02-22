from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict
import numpy as np
import sys
import os
def generate_input_vectors(num_vectors, dimension, on_probability, seed=1):

    n_active = int(on_probability*dimension)
    fname = 'vectors_{}_{}_{}_{}.npz'.format(num_vectors, dimension, n_active, seed)
    if os.path.isfile(fname):
        f = np.load(fname)
        return f['vectors']

    np.random.seed(seed)
    # vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    vecs = np.zeros((num_vectors, dimension))
    for i in range(num_vectors):
        indices = np.random.choice(np.arange(dimension, dtype='int'), size=n_active, replace=False)
        vecs[i, indices] = 1.0
    np.random.seed()

    np.savez_compressed(fname, vectors=vecs)

    return vecs

def generate_samples(input_vectors, num_samples, prob_noise, seed=1, method=None):
    """method='all' means randomly choose indices where we flip 1s and 0s with probability = prob_noise"""
    np.random.seed(seed)

    fname = 'samples_{}_{}_{}_{}.npz'.format(
        input_vectors.shape[0], input_vectors.shape[1], num_samples, seed)

    if os.path.isfile(fname):
        f = np.load(fname)
        return f['samples']

    samples = None

    for i in range(input_vectors.shape[0]):
        samp = np.tile(input_vectors[i, :], (num_samples, 1)).astype('int')
        if method == 'all':
            dice = np.random.uniform(0., 1., samp.shape)
            whr = np.where(dice < prob_noise)
            samp[whr] = 1 - samp[whr]
        elif method == 'exact':
            n_flips = int(np.mean(input_vectors.sum(axis=1)) * prob_noise)
            for j in range(num_samples):
                # flip zeros to ones
                indices = np.random.choice(np.where(samp[j] == 0)[0], size=n_flips, replace=False)
                samp[j, indices] = 1

                #flip ones to zeros
                indices = np.random.choice(np.where(samp[j] == 1)[0], size=n_flips, replace=False)
                samp[j, indices] = 0
        else:
            n_flips = int(np.mean(input_vectors.sum(axis=1)) * prob_noise) * 2
            for j in range(num_samples):
                indices = np.random.choice(np.arange(input_vectors.shape[1]), size=n_flips, replace=False)
                samp[j, indices] = 1 - samp[j, indices]

        if samples is None:
            samples = samp
        else:
            samples = np.append(samples, samp, axis=0)

    np.random.seed()

    np.savez_compressed(fname, samples=samples)

    return samples

def samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, seed=1,
    randomize_samples=False, regenerate=False):

    fname = 'spike_times_{}_{}_{}_{}_{}.npz'.format(
        samples.shape[0], samples.shape[1], sample_dt, start_dt, seed)

    if not regenerate and os.path.isfile(fname):
        f = np.load(fname)
        return f['indices'], f['spike_times'].tolist()

    np.random.seed(seed)
    t = 0
    spike_times = [[] for _ in range(samples.shape[-1])]
    if randomize_samples:
        indices = np.random.choice(np.arange(samples.shape[0]), size=samples.shape[0],
                    replace=False)
    else:
        indices = np.arange(samples.shape[0])

    total = float(len(indices))
    for i, idx in enumerate(indices):
        sys.stdout.write('\r\t\t%6.2f%%'%(100*((i+1.0)/total)))
        sys.stdout.flush()

        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        # max_start_dt = (sample_dt - start_dt)
        rand_start_dt = 0#np.random.randint(-start_dt, start_dt)
        ts = t + rand_start_dt + start_dt + np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size)
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt
    np.random.seed()

    sys.stdout.write('\n')
    sys.stdout.flush()

    np.savez_compressed(fname, spike_times=spike_times, indices=indices)

    return indices, spike_times
