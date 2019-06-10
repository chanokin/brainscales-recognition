import numpy as np
import os
# import sys
import glob

HEIGHT, WIDTH = 0, 1

def to_post(v, p, s, k):
    return ((v - k + 2 * p) // s) + 1

def randint_float(vmin, vmax):
    return np.float(np.random.randint(vmin, vmax))

def compute_num_regions(shape, stride, padding, kernel_shape):
    ins = np.array(shape)
    s = np.array(stride)
    p = np.array(padding)
    ks = np.array(kernel_shape)
    post_shape = to_post(ins, p, s, ks)
    return post_shape[0] * post_shape[1]

def n_neurons_per_region(num_in_layers, num_pi_divs):
    return num_in_layers * num_pi_divs

def n_in_gabor(shape, stride, padding, kernel_shape, num_in_layers, num_pi_divs):
    return compute_num_regions(shape, stride, padding, kernel_shape) * \
           n_neurons_per_region(num_in_layers, num_pi_divs)

def generate_input_vectors(num_vectors, dimension, on_probability, seed=1):
    n_active = int(on_probability*dimension)
    np.random.seed(seed)
    # vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    vecs = np.zeros((num_vectors, dimension))
    for i in range(num_vectors):
        indices = np.random.choice(np.arange(dimension, dtype='int'), size=n_active, replace=False)
        vecs[i, indices] = 1.0
    np.random.seed()
    return vecs

def generate_samples(input_vectors, num_samples, prob_noise, seed=1, method=None):
    """method='all' means randomly choose indices where we flip 1s and 0s with probability = prob_noise"""
    np.random.seed(seed)
    
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
    return samples

def samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, seed=1,
    randomize_samples=False):
    np.random.seed(seed)
    t = 0
    spike_times = [[] for _ in range(samples.shape[-1])]
    if randomize_samples:
        indices = np.random.choice(np.arange(samples.shape[0]), size=samples.shape[0],
                    replace=False)
    else:
        indices = np.arange(samples.shape[0])

    for idx in indices:
        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        ts = t + start_dt + np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size) 
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt
    np.random.seed()
    return indices, spike_times

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

def load_spike_file(dataset, digit, index):
    if dataset not in ['train', 't10k']:
        dataset = 'train'

    base_dir = "/home/gp283/brainscales-recognition/"\
               "codebase/NE15/mnist-db/spikes/"

    return sorted(glob.glob(
            os.path.join(base_dir, dataset, str(digit), '*.npz')))[index]




def pre_indices_per_region(pre_shape, pad, stride, kernel_shape):
    hk = np.array(kernel_shape) // 2

    pres = {}
    for _r in range(pad, pre_shape[HEIGHT], stride):
        post_r = to_post(_r, pad, stride, np.array(kernel_shape[HEIGHT])) - 1
        rdict = pres.get(post_r, {})
        for _c in range(pad, pre_shape[WIDTH], stride):
            post_c = to_post(_c, pad, stride, np.array(kernel_shape[WIDTH])) - 1
            clist = rdict.get(post_c, [])

            for k_r in range(-hk[HEIGHT], hk[HEIGHT] + 1, 1):
                for k_c in range(-hk[WIDTH], hk[WIDTH] + 1, 1):
                    pre_r, pre_c = _r + k_r, _c + k_c
                    outbound = pre_r < 0 or pre_c < 0 or \
                                pre_r >= pre_shape[HEIGHT] or \
                                pre_c >= pre_shape[WIDTH]

                    pre = None if outbound else (pre_r * pre_shape[WIDTH] + pre_c)
                    clist.append(pre)
            rdict[post_c] = clist
        pres[post_r] = rdict

    return pres


def kernel_pre_post_pairs(pre_shape, pad, stride, kernel_shape):
    post_shape = to_post(np.array(pre_shape), pad, stride, np.array(kernel_shape))
    hk = np.array(kernel_shape) // 2

    pairs = []
    for _r in range(pad, pre_shape[HEIGHT], stride):
        for _c in range(pad, pre_shape[WIDTH], stride):
            post_r = to_post(_r, pad, stride, np.array(kernel_shape[HEIGHT])) - 1
            post_c = to_post(_c, pad, stride, np.array(kernel_shape[WIDTH])) - 1
            for k_r in range(-hk[HEIGHT], hk[HEIGHT] + 1, 1):
                for k_c in range(-hk[WIDTH], hk[WIDTH] + 1, 1):
                    pre_r, pre_c = _r + k_r, _c + k_c
                    if pre_r < 0 or pre_c < 0 or \
                            pre_r >= pre_shape[HEIGHT] or \
                            pre_c >= pre_shape[WIDTH]:
                        continue

                    pre_id = pre_r * pre_shape[WIDTH] + pre_c
                    post_id = post_r * post_shape[WIDTH] + post_c
                    pairs.append([pre_id, post_id])

    return np.array(pairs)


def prob_conn_from_list(pre_post_pairs, n_per_post, probability, weight, delay, weight_off_mult=None):
    posts = np.unique(pre_post_pairs[:, 1])
    conns = []
    for post_base in posts:
        pres = pre_post_pairs[np.where(pre_post_pairs[:, 1] == post_base)]
        for i in range(n_per_post):
            for pre in pres:
                if np.random.uniform <= probability:
                    post = post_base * n_per_post + i
                    conns.append([pre, post, weight, delay])
                else:
                    if weight_off_mult is None:
                        continue

                    post = post_base * n_per_post + i
                    conns.append([pre, post, weight*weight_off_mult, delay])

    return np.array(conns)

def gabor_kernel(params):
    # adapted from
    # http://vision.psych.umn.edu/users/kersten/kersten-lab/courses/Psy5036W2017/Lectures/17_PythonForVision/Demos/html/2b.Gabor.html
    shape = np.array(params['shape'])
    omega = params['omega']  # amplitude1 (~inverse)
    theta = params['theta']  # rotation angle
    k = params.get('k', np.pi / 2.0)  # amplitude2
    sinusoid = params.get('sinusoid func', np.cos)
    normalize = params.get('normalize', True)

    r = np.floor(shape / 2.0)

    # create coordinates
    [x, y] = np.meshgrid(range(-r[0], r[0] + 1), range(-r[1], r[1] + 1))
    # rotate coords
    ct, st = np.cos(theta), np.sin(theta)
    x1 = x * ct + y * st
    y1 = x * (-st) + y * ct

    gauss = (omega ** 2 / (4.0 * np.pi * k ** 2)) * np.exp(
        (-omega ** 2 * (4.0 * x1 ** 2 + y1 ** 2)) * (1.0 / (8.0 * k ** 2)))
    sinus = sinusoid(omega * x1) * np.exp(k ** 2 / 2.0)
    k = gauss * sinus

    if normalize:
        k -= k.mean()
        k /= np.sqrt(np.sum(k ** 2))

    return k


def gabor_connect_templates(pre_indices, gabor_params, layer, delay=1.0):
    omegas = gabor_params['omega']
    omegas = omegas if isinstance(omegas, list) else [omegas]

    thetas = gabor_params['theta']
    thetas = thetas if isinstance(thetas, list) else [thetas]

    shape = gabor_params['shape']

    kernels = [gabor_kernel({'shape': shape, 'omega': o, 'theta': t})
               for o in omegas for t in thetas]

    conns = []
    for pre_i, pre in enumerate(pre_indices):
        if pre is None:
            continue

        r = pre_i // shape[1]
        c = pre_i % shape[1]
        for k in kernels:
            conns.append([pre, np.nan, k[r, c], delay])

    return kernels, conns


def split_spikes(spikes, n_types):
    spikes_out = [[] for _ in range(n_types)]
    n_per_type = spikes.shape[0] // n_types
    for type_idx in range(n_types):
        for nidx in range(n_per_type):
            spikes_out[type_idx].append(spikes[type_idx * n_per_type + nidx])
    return spikes_out


def div_index(orig_index, orig_shape, divs):
    r = (orig_index // orig_shape[1]) // divs[0]
    c = (orig_index % orig_shape[1]) // divs[1]
    return r * (orig_shape[1] // divs[1]) + c


def reduce_spike_place(spikes, shape, divs):
    fshape = [shape[0]//divs[0], shape[1]//divs[1]]
    fspikes = [[] for _ in range(fshape[0]*fshape[1])]
    for pre, times in spikes:
        fpre = div_index(pre, shape, divs)
        fspikes[fpre] += times
        fspikes[fpre][:] = np.unique(sorted(fspikes[fpre]))

    return fshape, fspikes


def scaled_pre_templates(pre_shape, pad, stride, kernel_shape, divs):
    pre_indices = []
    for scale_divs in divs:
        _indices = pre_indices_per_region(pre_shape, pad, stride, kernel_shape)
        if scale_divs[0] == 1 and scale_divs[1] == 1:
            pre_indices.append(_indices)
        else:
            d = {}
            for r in _indices:
                dr = d.get(r, {})
                for c in _indices[r]:
                    _scaled = set(dr.get(c, list()))
                    for pre in _indices[r][c]:
                        _scaled.add(div_index(pre, pre_shape, scale_divs))
                    dr[c] = list(_scaled)
                d[r] = dr

            pre_indices.append(d)

    return pre_indices