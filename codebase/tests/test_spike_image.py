from __future__ import print_function
import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from scipy import misc as scipy_misc
from scipy import ndimage
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
from spikevo.image_input import SpikeImage, CHAN2COLOR, CHAN2TXT
from pprint import pprint 

POISSON_SOURCES = bool(0)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--backend', help='available %s'%supported_backends, 
    default=GENN)
parser.add_argument('image_path', help='where to locate an input image',
    )

args = parser.parse_args()
image_path = args.image_path
backend = args.backend
pynn = backend_setup(backend)
pynnx = PyNNAL(pynn)


image = ndimage.imread(image_path, flatten=True).astype('float32')
image /= 255.0
plt.figure()
plt.suptitle('input image')
plt.imshow(image)
# plt.show()

height, width = image.shape

N_NEURONS = height*width
w = 0.025
syn_delay = 1.
sim_time = 100.

neuron_parameters = {
    'v_thresh':   -35.0, 
    'tau_m':       20.,
    'tau_syn_E':   10.0, 
    # 'e_rev_E':     0., 
    'tau_refrac':  0.1 , 
    'v_reset':    -50.0,  #hdbrgs
    'tau_syn_I':   5., 
    'i_offset':    0.0,
    #ESS - BrainScaleS
    'cm':          0.2,
    'v_rest':     -50.0,
    # 'e_rev_I':    -100.,
} 

pynn.setup(timestep=1.0, min_delay=1.0)

spk_image = SpikeImage(width, height, 
                encoding_params={'rate': 1000, 'threshold': 0.5})
spike_rates = spk_image.encode(image)
image_spikes = {ch: spk_image.rate_to_poisson(spike_rates[ch], 0, sim_time) \
                                                        for ch in spike_rates}
if POISSON_SOURCES:
    image_pops = spk_image.create_pops(pynnx, 
                    {ch: {'rate': spike_rates[ch] }for ch in spike_rates},
                    'poisson')
else:
    image_pops = spk_image.create_pops(pynnx, 
                {ch: {'spike_times': image_spikes[ch] }for ch in image_spikes})

rate_render = np.zeros((height, width, 3))
for ch in spike_rates:
    c = CHAN2COLOR[ch]
    for i, r in enumerate(spike_rates[ch]):
        row, col = i//width, i%width
        if r > 0:
            print(CHAN2TXT[ch], row, col)
        rate_render[row, col, c] = r
        
plt.figure()
plt.suptitle('rate image')
plt.imshow(rate_render)

spike_render = np.zeros((height, width, 3))
for ch in spike_rates:
    c = CHAN2COLOR[ch]
    for i, times in enumerate(image_spikes[ch]):
        row, col = i//width, i%width
        for t in times:
            spike_render[row, col, c] += 1
        
plt.figure()
plt.suptitle('spike image')
plt.imshow(spike_render)

neurons = {}
projs = {}
for ch in image_pops:
    image_pops[ch].record('spikes')
    neurons[ch] = pynnx.Pop(N_NEURONS, pynn.IF_cond_exp, neuron_parameters)
    neurons[ch].record('spikes')
    projs[ch] = pynnx.Proj(image_pops[ch], neurons[ch], pynn.OneToOneConnector, 
        weights=w, delays=syn_delay, label="source to neurons {}".format(ch))

pynn.run(sim_time)

in_spikes = {}
for ch in image_pops:
    in_spikes[ch] = pynnx.get_spikes(image_pops[ch])

out_spikes = {}
for ch in image_pops:
    out_spikes[ch] = pynnx.get_spikes(neurons[ch])

pynn.end()


fig = plt.figure()
plt.suptitle('input source spikes')
ax = plt.subplot(1, 1, 1)
spike_render = spk_image.spikes_to_image(in_spikes)
plt.imshow(spike_render)
ax.set_xlabel('Post id')
ax.set_ylabel('Pre id')


fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
spike_render = spk_image.spikes_to_image(out_spikes)
plt.imshow(spike_render)
ax.set_xlabel('Post id')
ax.set_ylabel('Pre id')

plt.savefig("output.pdf")
plt.show()




