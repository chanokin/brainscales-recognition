from __future__ import (print_function,
                        unicode_literals,
                        division)
import os
from glob import glob
import numpy as np
import pynn_genn as sim
from pprint import pprint
from config import *
from utils import *
import matplotlib.pyplot as plt

class Decoder(object):
    def __init__(self, name, params):
        self._network = None
        self.name = name
        self.decode(params)
        self.inputs = None
        print("In Decoder init, %s"%name)

    def decode(self, params):
        net = {}
        net['timestep'] = params.get('timestep', TIMESTEP)
        net['min_delay'] = params.get('min_delay', TIMESTEP)
        net['run_time'] = 1.0#params['run_time']

        self.inputs = self.get_in_spikes(params)

        base = 0
        plt.figure()
        for i in self.inputs:

            for n, times in enumerate(self.inputs[i]):
                plt.plot(times, base + n*np.ones_like(times), '.b', markersize=1)
            base += n

        plt.show()

        pops = {}
        projs = {}

        net['populations'] = pops
        net['projections'] = projs

        pprint(params)

        self._network = net

    def get_in_spikes(self, params):
        path = params['sim']['spikes_path']
        nclass = params['sim']['num_classes']
        nsamp = params['sim']['samples_per_class']
        nlayers = params['sim']['input_layers']
        in_shape = params['sim']['input_shape']
        dt = params['sim']['sample_dt']

        fnames = []
        for cidx in range(nclass):
            cpath = os.path.join(path, str(cidx))
            files = glob.glob(os.path.join(cpath, '*.npz'))
            for f in files[:nsamp]:
                fnames.append(f)

        from random import shuffle
        shuffle(fnames)
        pprint(fnames)

        tmp = []
        spikes = {i: None for i in range(nlayers)}
        dt_idx = 0
        for f in fnames:
            spk = np.load(f, allow_pickle=True)
            tmp[:] = split_spikes(spk['spikes'], nlayers)
            for tidx in range(nlayers):
                divs = (1, 1) if tidx < 2 else params['sim']['input_divs']
                shape, tmp[tidx][:] = reduce_spike_place(tmp[tidx], in_shape, divs)
                if spikes[tidx] is None:
                    spikes[tidx] = tmp[tidx]
                else:
                    spikes[tidx][:] = append_spikes(spikes[tidx], tmp[tidx], dt_idx*dt)

            dt_idx += 1

        return spikes




    def run_pynn(self):
        net = self._network
        pprint(net)
        # sim.setup(net['timestep'], net['min_delay'],
        #           model_name=self.name,
        #           backend='SingleThreadedCPU'
        #           )
        for pop in net['populations']:
            pass

        for proj in net['projections']:
            pass

        # sim.run(net['run_time'])

        records = {}
        for pop in net['populations']:
            pass

        weights = {}
        for proj in net['projections']:
            pass

        # sim.end()


        data = {'recs': records, 'weights': weights}
        return data