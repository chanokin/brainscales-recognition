from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import sys
from matplotlib import gridspec
import argparse
import os


def plot_snap(start_t, end_t, volts, spikes_in, spikes_out, ts=1.0, start_neuron=0, end_neuron=-1):
    ms_to_ts = 1.0 / ts
    ts_to_ms = 1.0 / ms_to_ts
    ts_start = int(start_t * ms_to_ts)
    ts_end = int(end_t * ms_to_ts)
    start_nid = start_neuron
    end_nid = end_neuron

    post_spikes = []
    post_ids = []
    for i in range(start_nid, end_nid):
        local_spikes = np.array(spikes_out[i])
        whr = np.where(np.logical_and(
            start_t <= local_spikes,
            local_spikes < end_t))
        post_spikes.append(local_spikes[whr])
        if len(whr[0]):
            post_ids.append(i)
    print(post_ids)

    pre_spikes = []
    for i in range(len(spikes_in)):
        local_spikes = np.array(spikes_in[i])
        whr = np.where(np.logical_and(
            start_t <= local_spikes,
            local_spikes < end_t))
        pre_spikes.append(local_spikes[whr])

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    for i, times in enumerate(pre_spikes):
        for t in times:
            plt.axvline((t - ts_to_ms - start_t) * ms_to_ts, linestyle='--', linewidth=0.5, color='magenta')

    for i, times in enumerate(post_spikes):
        for t in times:
            plt.axvline((t - ts_to_ms - start_t) * ms_to_ts, marker='.', linestyle='--', linewidth=0.5, color='blue')

    plt.plot(volts[ts_start:ts_end, start_nid:end_nid])
    plt.axhline(-65, color='black')
    xticks = ax.get_xticks()
    ax.set_xticklabels(start_t + np.array(xticks) * ts_to_ms)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('membrane voltage [mV]')

    plt.show()


def plot_in_range(fig, ax, spikes, start_t, end_t, color='blue', markersize=1, marker='.', markeredgewidth=0,
                  plot_line=None, linewidth=0):
    for i, times in enumerate(spikes):
        times = np.array(times)
        whr = np.where(np.logical_and(times >= start_t, times <= end_t))[0].astype('int')
        if len(whr):
            plot_times = times[whr]
            if plot_line is not None:
                for t in plot_times:
                    if plot_line == 'vertical':
                        plt.axvline(t, color=color, linewidth=linewidth)
                    elif plot_line == 'horizontal':
                        plt.axhline(t, color=color, linewidth=linewidth)
            else:
                plt.plot(plot_times, i * np.ones_like(plot_times), marker, color=color,
                         markersize=markersize, markeredgewidth=markeredgewidth)


def active_per_pattern(spikes, dt=50):
    end = 0
    n_active = []

    for start in range(0, total_t, dt):
        sys.stdout.write("\r{}/{}".format(start, total_t))
        sys.stdout.flush()
        end = start + dt
        active_count = 0
        for times in spikes:

            times = np.array(times)
            whr = np.where(np.logical_and(times >= start, times < end))[0]
            if len(whr):
                active_count += 1

        n_active.append(active_count)

    sys.stdout.write("\n")
    sys.stdout.flush()

    return n_active


def pop_rate_per_second(spikes, total_t, dt):
    ms_to_s = 1. / 1000.
    n_neurons = len(spikes)
    end = 0
    rates = []
    for start in range(0, total_t, dt):
        end = start + dt
        spike_count = 0
        for times in spikes:
            times = np.array(times)
            whr = np.where(np.logical_and(times >= start, times < end))[0]
            spike_count += len(whr)
        rate = float(spike_count) / float(n_neurons * dt * ms_to_s)
        #         rate = float(spike_count) / float(dt * ms_to_s)
        rates.append(rate)

    return rates


def active_neurons_per_pattern(spikes, start_time, end_time, sample_indices, start_indices_index, config):
    t_per_sample = config['time_per_sample']
    n_samples = config['n_samples']
    n_patterns = config['n_patterns']

    posts = []
    active_neurons = {idx: set() for idx in range(n_patterns)}
    st = start_time
    for idx in range(start_indices_index, len(sample_indices)):
        pat_id = sample_indices[idx] // n_samples
        et = st + t_per_sample
        posts[:] = []
        for post_id, times in enumerate(spikes):
            times = np.array(times)
            find = np.where(np.logical_and(times >= st, times < et))[0]
            if len(find):
                posts.append(post_id)

        #         print(pat_id, st, et, posts)
        ### update set of active neurons
        active_neurons[pat_id] |= set(posts)

        st = et

    return active_neurons


def avg_mean(sig):
    return np.sum(sig) / float(len(sig))


def energy(sig):
    return np.sum(np.power(sig, 2))


def power(sig):
    return energy(sig) / float(len(sig))


def variance(sig):
    return power(sig) - avg_mean(sig) ** 2

def analyse_spike_patterns(input_spikes, input_vecs, sample_indices, total_t,
                           sample_dt, start_t=0, do_overlaps=True):
    n_patterns = len(input_vecs)
    pop_size = len(input_spikes)
    n_samples = len(sample_indices)//n_patterns
    if do_overlaps:
        all_overlaps = np.zeros((n_patterns, n_samples))
    else:
        all_overlaps = np.zeros((n_patterns, 1))

    start_idx = int(start_t//sample_dt)
    sample_idx = np.zeros(n_patterns)
    patterns = np.zeros((n_patterns, pop_size))
    for i, t in enumerate(np.arange(start_t, total_t, sample_dt)):
        sys.stdout.write('\r%6.2f%%' % (float((i + 1) * 100) / float(n_samples * n_patterns)))
        sys.stdout.flush()
        sid = start_idx + i
        curr_pat = int(sample_indices[sid] // n_samples)

        spikes = np.zeros(pop_size)
        for j in range(pop_size):
            row = np.array(input_spikes[j])
            whr = np.where(np.logical_and(t <= row, row < t + sample_dt))
            if len(whr[0]):
                spikes[j] = 1

        patterns[curr_pat, :] += spikes

        if do_overlaps:
            overlap = np.logical_and(input_vecs[curr_pat], spikes).sum()
            all_overlaps[curr_pat, int(sample_idx[curr_pat])] = overlap
            sample_idx[curr_pat] += 1

    sys.stdout.write('\n')
    sys.stdout.flush()

    return patterns, all_overlaps

def plot_weight_figs(initial, final, difference, out_suffix, cmap='seismic_r'):
    max_end = np.max(np.abs(final))
    max_start = np.max(np.abs(initial))
    max_diff = np.abs(difference).max()

    max_w = max(max_diff, max(max_start, max_end))

    fig = plt.figure(figsize=(20, 2))
    ax = plt.subplot(1, 1, 1)
    im = plt.imshow(initial.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("weight_initial_{}.pdf".format(out_suffix))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 2))
    ax = plt.subplot(1, 1, 1)
    im = plt.imshow(final.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("weight_final_{}.pdf".format(out_suffix))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 2))
    ax = plt.subplot(1, 1, 1)
    im = plt.imshow(difference.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("weight_difference_{}.pdf".format(out_suffix))
    plt.close(fig)


    fig = plt.figure(figsize=(20, 5))
    ax = plt.subplot(3, 1, 1)
    im = plt.imshow(initial.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    ax.set_title('Initial')

    ax = plt.subplot(3, 1, 2)
    im = plt.imshow(final.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    ax.set_title('Final')

    ax = plt.subplot(3, 1, 3)
    im = plt.imshow(difference.transpose(),
                    cmap=cmap, vmin=-max_w, vmax=max_w)
    plt.colorbar(im, ax=ax)
    ax.set_title('Difference')

    plt.tight_layout()

    plt.savefig("weight_change_{}.pdf".format(out_suffix))
    # plt.show()
    plt.close(fig)

CMAP = 'nipy_spectral'
# CMAP = 'gist_stern'
# CMAP = 'inferno'
# CMAP = 'magma'
# CMAP = 'plasma'
# CMAP = 'ocean'


def plot_vector_distances(vectors, cmap=CMAP):
    n_vectors = len(vectors)
    angles = np.zeros((n_vectors, n_vectors))
    dots = np.zeros((n_vectors, n_vectors))
    weights = np.zeros((n_vectors, n_vectors))
    divs = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        a = vectors[i]
        na = np.sqrt(np.dot(a, a))
        for j in range(n_vectors):
            b = vectors[j]
            nb = np.sqrt(np.dot(b, b))

            dots[i, j] = np.dot(a, b)
            weights[i, j] = (na * nb)
            divs[i, j] = dots[i, j] / weights[i, j]
            angles[i, j] = np.rad2deg( np.arccos( divs[i, j] ) )

    angles[np.isclose(divs, 1.0)] = 0.0

    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(1, 1, 1)
    im = plt.imshow(angles, interpolation='none', cmap=cmap, vmin=0, vmax=90)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    xticks = ax.get_xticks().astype('int')
    ax.set_xticklabels(xticks + 1)
    yticks = ax.get_yticks().astype('int')
    ax.set_yticklabels(yticks + 1)

    return fig, ax


def plot_input_vector_distance(_vectors, out_dir, out_suffix, cmap=CMAP):
    fig, ax = plot_vector_distances(_vectors, cmap)
    # ax.set_title('Angle between input vectors (deg)')
    fname = os.path.join(out_dir, 'input_vector_distances_{}.pdf'.format(out_suffix))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


def plot_input_spikes_distance(_vectors, out_dir, out_suffix, cmap=CMAP):
    fig, ax = plot_vector_distances(_vectors, cmap)
    # ax.set_title('Angle between input spikes (deg)')
    fname = os.path.join(out_dir, 'noisy_input_spike_distances_{}.pdf'.format(out_suffix))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


def plot_kenyon_spikes_distance(_vectors, out_dir, out_suffix, cmap=CMAP):
    fig, ax = plot_vector_distances(_vectors, cmap)
    # ax.set_title('Angle between kenyon neurons spikes (deg)')
    fname = os.path.join(out_dir, 'kenyon_spike_distances_{}.pdf'.format(out_suffix))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


def plot_start_decision_spikes_distance(_vectors, out_dir, out_suffix, cmap=CMAP):
    fig, ax = plot_vector_distances(_vectors, cmap)
    # ax.set_title('Angle between decision neurons spikes (start; deg)')
    fname = os.path.join(out_dir, 'decision_start_spike_distances_{}.pdf'.format(out_suffix))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


def plot_end_decision_spikes_distance(_vectors, out_dir, out_suffix, cmap=CMAP):
    fig, ax = plot_vector_distances(_vectors, cmap)
    # ax.set_title('Angle between decision neurons spikes (end; deg)')
    fname = os.path.join(out_dir, 'decision_end_spike_distances_{}.pdf'.format(out_suffix))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



parser = argparse.ArgumentParser(description='Mushroom body experiment analysis')
parser.add_argument('filename', type=str, default='', help='filename to analyze')

this_args = parser.parse_args()
fname = this_args.filename
out_suffix = os.path.basename(os.path.normpath(fname))[:-4]
out_dir = os.path.dirname(os.path.normpath(fname))
out_fname = os.path.join(out_dir, 'analysis___{}.npz'.format(out_suffix))
if os.path.isfile(out_fname):
    analysis = np.load(out_fname)
else:
    analysis = None

data = np.load(fname)
args = data['args'].item()
thr = 0.25

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
sys.stdout.write("Rendering weights and their difference\n")
sys.stdout.flush()

end_w = data['output_end_weights'].reshape((args.nKC, args.nDN))
start_conn = data['output_start_connections']
start_w = np.zeros((args.nKC, args.nDN))
for row in start_conn:
    pre, post, w, d = row[0], row[1], row[2], row[3]
    start_w[int(pre), int(post)] = w

diff_w = end_w - start_w

plot_weight_figs(start_w, end_w, diff_w, out_suffix)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
sys.stdout.write("Rendering input vectors distances\n")
sys.stdout.flush()

input_vecs = data['input_vectors']
plot_input_vector_distance(input_vecs, out_dir, out_suffix)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# sys.stdout.write("Rendering input spike distances\n")
# sys.stdout.flush()
#
in_spikes = data['input_spikes']
total_t = int(data['sim_time'])
sample_indices = data['sample_indices']
n_samples = args.nSamplesAL
n_patterns = args.nPatternsAL
size_al = args.nAL
sample_dt = data['sample_dt']
in_patterns_union = []
in_patterns_overlap = []
# if analysis is None:
#     in_patterns_union, in_patterns_overlap = \
#         analyse_spike_patterns(in_spikes, input_vecs, sample_indices, total_t,
#                                sample_dt)
# else:
#     in_patterns_union = analysis['in_patterns_union']
#     in_patterns_overlap = analysis['in_patterns_overlap']
#
# tmp = (in_patterns_union >= (n_samples * thr)).astype('float')
# plot_input_spikes_distance(tmp, out_dir, out_suffix)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
sys.stdout.write("Rendering kenyon spike distances\n")
sys.stdout.flush()

k_spikes = data['kenyon_spikes']
if analysis is None:
    k_patterns_union, k_patterns_overlap = \
        analyse_spike_patterns(k_spikes, input_vecs, sample_indices, total_t,
                               sample_dt, do_overlaps=False)
else:
    k_patterns_union = analysis['k_patterns_union']
    k_patterns_overlap = analysis['k_patterns_overlap']

tmp = (k_patterns_union >= (n_samples * thr)).astype('float')
plot_kenyon_spikes_distance(tmp, out_dir, out_suffix)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
sys.stdout.write("Rendering decision spike distances\n")
sys.stdout.flush()

d_spikes = data['decision_spikes']
n_test = data['n_test_samples']
end_t = n_test * float(data['sample_dt'])
if analysis is None:
    d_start_patterns_union, d_start_patterns_overlap = \
        analyse_spike_patterns(d_spikes, input_vecs, sample_indices, end_t,
                               sample_dt, do_overlaps=False)
else:
    d_start_patterns_union = analysis['d_start_patterns_union']
    d_start_patterns_overlap = analysis['d_start_patterns_overlap']

tmp = (d_start_patterns_union > 0).astype('float')
plot_start_decision_spikes_distance(tmp, out_dir, out_suffix)

start_idx = max(0, args.nPatternsAL * args.nSamplesAL - (n_test) )
start_t = int(start_idx * float(data['sample_dt']))
if analysis is None:
    d_end_patterns_union, d_end_patterns_overlap = \
        analyse_spike_patterns(d_spikes, input_vecs, sample_indices, total_t,
                               sample_dt, start_t=start_t, do_overlaps=False)
else:
    d_end_patterns_union = analysis['d_end_patterns_union']
    d_end_patterns_overlap = analysis['d_end_patterns_overlap']

tmp = (d_end_patterns_union > 0).astype('float')
plot_end_decision_spikes_distance(tmp, out_dir, out_suffix)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


np.savez_compressed(out_fname,
    in_patterns_union=in_patterns_union,
    in_patterns_overlap=in_patterns_overlap,
    k_patterns_union=k_patterns_union,
    k_patterns_overlap=k_patterns_overlap,
    d_start_patterns_union=d_start_patterns_union,
    d_start_patterns_overlap=d_start_patterns_overlap,
    d_end_patterns_union=d_end_patterns_union,
    d_end_patterns_overlap=d_end_patterns_overlap,
)