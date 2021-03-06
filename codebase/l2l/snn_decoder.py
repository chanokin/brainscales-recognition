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
import sys
from analyse_network import spiking_per_class

if DEBUG:
    class Logging:
        def __init__(self):
            pass
        def info(self, txt):
            sys.stdout.write(str(txt)+'\n')
            sys.stdout.flush()

    logging = Logging()
else:
    import logging

class Decoder(object):
    def __init__(self, name, params):
        self._network = None
        self.inputs = None
        self.in_shapes = None
        self.in_labels = None
        self.name = name
        self.params = params
        self.decode(params)
        logging.info("In Decoder init, %s"%name)
        # pprint(params)

    def decode(self, params):
        self._network = {}
        self._network['timestep'] = params['sim'].get('timestep', TIMESTEP)
        self._network['min_delay'] = params['sim'].get('min_delay', TIMESTEP)
        self._network['run_time'] = params['sim']['duration']

        logging.info("Setting up simulator")
        sim.setup(self._network['timestep'],
                  self._network['min_delay'],
                  model_name=self.name,
                  backend='SingleThreadedCPU',
                )

        logging.info("\tGenerating spikes")
        self.in_labels, self.in_shapes, self.inputs = self.get_in_spikes(params)

        pops = {}
        logging.info("\tPopulations: Input")
        pops['input'] = self.input_populations()

        logging.info("\tPopulations: Gabor")
        self.gabor_shapes, pops['gabor'] = self.gabor_populations(params)

        logging.info("\tPopulations: Mushroom")
        pops['mushroom'] = self.mushroom_population(params)

        logging.info("\tPopulations: Mushroom Inhibitory")
        pops['inh_mushroom'] = self.inh_mushroom_population(params)

        logging.info("\tPopulations: Output")
        pops['output'] = self.output_population(params)

        logging.info("\tPopulations: Output Inhibitory")
        pops['inh_output'] = self.inh_output_population(params)

        self._network['populations'] = pops


        projs = {}
        logging.info("\tProjections: Input to Gabor")
        projs['input to gabor'] = self.in_to_gabor(params)

        logging.info("\tProjections: Gabor to Mushroom")
        projs['gabor to mushroom'] = self.gabor_to_mushroom(params)

        logging.info("\tProjections: Mushroom to Output")
        projs['mushroom to output'] = self.mushroom_to_output(params)

        logging.info("\tProjections: Gabor sWTA")
        projs['wta_mushroom'] = self.wta_gabor(params)

        logging.info("\tProjections: Mushroom sWTA")
        projs['wta_mushroom'] = self.wta_mushroom(params)

        logging.info("\tProjections: Output sWTA")
        projs['wta_output'] = self.wta_output(params)



        self._network['projections'] = projs


    def get_in_spikes(self, params):
        path = params['sim']['spikes_path']
        nclass = params['sim']['num_classes']
        nsamp = params['sim']['total_per_class']
        nlayers = params['sim']['input_layers']
        in_shape = params['sim']['input_shape']
        dt = params['sim']['sample_dt']
        if DEBUG and bool(0):
            return ([0], {0: [28, 28], 1: [28, 28], 2: [6, 10], 3: [6, 10]},\
           {0: [[np.round(np.random.uniform(0, 10.0), decimals=1)] if np.random.uniform(0,1) > 0.75 else [] for _ in range(28 * 28)],
            1: [[np.round(np.random.uniform(0, 10.0), decimals=1)] if np.random.uniform(0,1) > 0.75 else [] for _ in range(28 * 28)],
            2: [[np.round(np.random.uniform(0, 10.0), decimals=1)] if np.random.uniform(0,1) > 0.75 else [] for _ in range(10 * 6)],
            3: [[np.round(np.random.uniform(0, 10.0), decimals=1)] if np.random.uniform(0,1) > 0.75 else [] for _ in range(10 * 6)]})

        fnames = []
        for cidx in range(nclass):
            cpath = os.path.join(path, str(cidx))
            files = glob.glob(os.path.join(cpath, '*.npz'))
            for f in files[:nsamp]:
                fnames.append(f)

        from random import shuffle
        shuffle(fnames)

        tmp = []
        labels = []
        spikes = {i: None for i in range(nlayers)}
        shapes = {i: None for i in range(nlayers)}
        dt_idx = 0
        for f in fnames:
            spk = np.load(f, allow_pickle=True)
            labels.append(spk['label'].item())
            tmp[:] = split_spikes(spk['spikes'], nlayers)
            for tidx in range(nlayers):
                divs = (1, 1) if tidx < 2 else params['sim']['input_divs']
                shape, tmp[tidx][:] = reduce_spike_place(tmp[tidx], in_shape, divs)
                if shapes[tidx] is None:
                    shapes[tidx] = shape
                if spikes[tidx] is None:
                    spikes[tidx] = tmp[tidx]
                else:
                    spikes[tidx][:] = append_spikes(spikes[tidx], tmp[tidx], dt_idx*dt)

            dt_idx += 1

        return labels, shapes, spikes


    ### ----------------------------------------------------------------------
    ### -----------------          populations           ---------------------
    ### ----------------------------------------------------------------------

    def input_populations(self):
        if self.inputs is None:
            raise Exception("Input spike arrays are not defined")

        try:
            return self._network['populations']['input']
        except:

            ins = {}
            for i in self.inputs:
                s = len(self.inputs[i])
                p = sim.Population(s, sim.SpikeSourceArray,
                                   {'spike_times': self.inputs[i]},
                                   label='input layer %s'%i)
                if 'input' in record_spikes:
                    p.record('spikes')
                ins[i] = p
            return ins

    def gabor_populations(self, params=None):
        try:
            return self.gabor_shapes, self._network['populations']['gabor']
        except:
            gs = {}
            stride = (params['ind']['stride'], params['ind']['stride'])
            pad = (params['sim']['kernel_pad'], params['sim']['kernel_pad'])
            k_shape = (params['sim']['kernel_width'], params['sim']['kernel_width'])
            ndivs = int(params['ind']['n_pi_divs'])
            _shapes = {
                i: compute_region_shape(self.in_shapes[i], stride, pad, k_shape) \
                                                            for i in self.in_shapes
            }
            neuron_type = getattr(sim, gabor_class)
            for lyr in _shapes:
                lrd = gs.get(lyr, {})
                for row in np.arange(_shapes[lyr][HEIGHT]).astype('int'):
                    lrc = lrd.get(row, {})
                    for col in np.arange(_shapes[lyr][WIDTH]).astype('int'):
                        lrc[col] = sim.Population(ndivs, neuron_type, gabor_params,
                                    label='gabor - {} ({}, {})'.format(lyr, row, col))
                        if 'gabor' in record_spikes:
                            lrc[col].record('spikes')

                    lrd[row] = lrc
                gs[lyr] = lrd

            return _shapes, gs

    def mushroom_population(self, params=None):
        try:
            return self._network['populations']['mushroom']
        except:
            expand = params['ind']['expand']
            ndivs = int(params['ind']['n_pi_divs'])

            gshapes = self.gabor_shapes
            count = 0
            for l in gshapes:
                count += int(gshapes[l][0]*gshapes[l][1]*ndivs)
            count = int(count * expand)
            neuron_type = getattr(sim, mushroom_class)
            p = sim.Population(count, neuron_type, mushroom_params,
                               label='mushroom')

            if 'mushroom' in record_spikes:
                p.record('spikes')

            return p

    def inh_mushroom_population(self, params=None):
        try:
            return self._network['populations']['inh_mushroom']
        except:
            neuron_type = getattr(sim, inh_mushroom_class)
            p = sim.Population(1, neuron_type, inh_mushroom_params,
                               label='inh_mushroom')

            if 'inh_mushroom' in record_spikes:
                p.record('spikes')

            return p


    def output_population(self, params=None):
        try:
            return self._network['populations']['output']
        except:
            neuron_type = getattr(sim, output_class)
            n_out = params['sim']['output_size']
            p = sim.Population(n_out, neuron_type, output_params,
                               label='output')

            if 'output' in record_spikes:
                p.record('spikes')

            return p

    def inh_output_population(self, params=None):
        try:
            return self._network['populations']['inh_output']
        except:
            neuron_type = getattr(sim, inh_output_class)
            p = sim.Population(1, neuron_type, inh_output_params,
                               label='inh_output')

            if 'inh_output' in record_spikes:
                p.record('spikes')

            return p


    ### ----------------------------------------------------------------------
    ### -----------------          projections           ---------------------
    ### ----------------------------------------------------------------------

    def in_to_gabor(self, params=None):
        try:
            return self._network['projections']['input to gabor']
        except:
            gabor_weight = [params['ind'][s] for s in sorted(params['ind'].keys()) \
                                                        if s.startswith('gabor_weight')]
            stride = (int(params['ind']['stride']), int(params['ind']['stride']))
            pad = (int(params['sim']['kernel_pad']), int(params['sim']['kernel_pad']))
            k_shape = (int(params['sim']['kernel_width']), int(params['sim']['kernel_width']))
            ndivs = params['ind']['n_pi_divs']
            # 0 about the same as 180?
            adiv = (np.pi / ndivs)

            gabor_params = {
                'omega': [params['ind']['omega']],
                'theta': (np.arange(ndivs) * adiv).tolist(),
                'shape': k_shape,
            }
            pre_shapes = self.in_shapes
            pres = self.input_populations()
            post_shapes, posts = self.gabor_populations()

            projs = {}
            for i in self.in_shapes:
                lyrdict = projs.get(i, {})
                pre_shape = pre_shapes[i]
                pre_indices = pre_indices_per_region(pre_shape, pad, stride, k_shape)
                pre = pres[i]
                for r in posts[i]:
                    rowdict = lyrdict.get(r, {})
                    for c in posts[i][r]:
                        k, conns = gabor_connect_list(pre_indices[r][c], gabor_params, delay=1.0,
                                                      w_mult=gabor_weight[i])
                        ilist, elist = split_to_inh_exc(conns)

                        if len(elist) == 0:
                            continue

                        post = posts[i][r][c]
                        econ = sim.FromListConnector(elist)
                        elabel = 'exc - in {} to gabor {}-{}'.format(i, r, c)
                        rowdict[c] = {
                            'exc': sim.Projection(pre, post, econ, sim.StaticSynapse(),
                                                  label=elabel, receptor_type='excitatory')
                        }

                        if len(ilist) > 0:
                            icon = sim.FromListConnector(ilist)
                            ilabel = 'inh - in {} to gabor {}-{}'.format(i, r, c)
                            rowdict[c]['inh'] = sim.Projection(pre, post, icon, sim.StaticSynapse(),
                                                               label=ilabel, receptor_type='inhibitory')

                    lyrdict[r] = rowdict
                projs[i] = lyrdict

            return projs

    def gabor_to_mushroom(self, params=None):
        try:
            return self._network['projections']['gabor to mushroom']
        except:
            post = self.mushroom_population()
            prob = params['ind']['exp_prob']
            mushroom_weight = params['ind']['mushroom_weight']
            projs = {}
            pre_shapes, pres = self.gabor_populations()
            for lyr in pres:
                lyrdict = projs.get(lyr, {})
                for r in pres[lyr]:
                    rdict = lyrdict.get(r, {})
                    for c in pres[lyr][r]:
                        pre = pres[lyr][r][c]
                        rdict[c] = sim.Projection(pre, post,
                                    sim.FixedProbabilityConnector(prob),
                                    sim.StaticSynapse(weight=mushroom_weight),
                                   label='gabor {}{}{} to mushroom'.format(lyr, r, c),
                                    receptor_type='excitatory')
                    lyrdict[r] = rdict
                projs[lyr] = lyrdict

            return projs

    def wta_gabor(self, params=None):
        try:
            return self._network['projections']['wta_gabor']
        except:
            iw = inhibitory_weight
            projs = {}
            pre_shapes, pres = self.gabor_populations()
            for lyr in pres:
                lyrdict = projs.get(lyr, {})
                for r in pres[lyr]:
                    rdict = lyrdict.get(r, {})
                    for c in pres[lyr][r]:
                        pre = pres[lyr][r][c]
                        post = pres[lyr][r][c]

                        rdict[c] = sim.Projection(pre, post,
                                    sim.AllToAllConnector(),
                                    sim.StaticSynapse(weight=iw, delay=TIMESTEP),
                                   label='wta gabor {}{}{}'.format(lyr, r, c),
                                    receptor_type='inhibitory')
                    lyrdict[r] = rdict
                projs[lyr] = lyrdict

            return projs


    def wta_mushroom(self, params=None):
        try:
            return self._network['projections']['wta_mushroom']
        except:
            prjs = {}
            pre_size = np.sum([s[0]*s[1] for i, s in self.gabor_shapes.items()])
            exc = self.mushroom_population()
            inh = self.inh_mushroom_population()
            ew = excitatory_weight['mushroom'] * (pre_size / float(exc.size))

            prjs['e to i'] = sim.Projection(exc, inh,
                                sim.AllToAllConnector(),
                                sim.StaticSynapse(weight=ew, delay=TIMESTEP),
                                label='mushroom to inh_mushroom',
                                receptor_type='excitatory')

            prjs['i to e'] = sim.Projection(inh, exc,
                                sim.AllToAllConnector(),
                                sim.StaticSynapse(weight=inhibitory_weight, delay=TIMESTEP),
                                label='inh_mushroom to mushroom',
                                receptor_type='inhibitory')

            return prjs


    def mushroom_to_output(self, params=None):
        try:
            return self._network['projections']['mushroom to output']
        except:
            pre = self.mushroom_population()
            post = self.output_population()
            prob = params['ind']['out_prob']
            max_w = params['ind']['out_weight']
            conn_list = output_connection_list(pre.size, post.size, prob,
                           max_w, 0.1, seed=123)
            tdep = getattr(sim, time_dep)(tau_plus, tau_minus, A_plus, A_minus)
            wdep = getattr(sim, weight_dep)(w_min_mult*max_w, w_max_mult*max_w)
            stdp = sim.STDPMechanism(timing_dependence=tdep, weight_dependence=wdep)

            p = sim.Projection(pre, post, sim.FromListConnector(conn_list), stdp,
                               label='mushroom to output', receptor_type='excitatory')
            return p


    def wta_output(self, params=None):
        try:
            return self._network['projections']['wta_output']
        except:
            prjs = {}
            exc = self.output_population()
            inh = self.inh_output_population()
            ew = excitatory_weight['output']

            prjs['e to i'] = sim.Projection(exc, inh,
                                sim.AllToAllConnector(),
                                sim.StaticSynapse(weight=ew, delay=TIMESTEP),
                                label='output to inh_output',
                                receptor_type='excitatory')

            prjs['i to e'] = sim.Projection(inh, exc,
                                sim.AllToAllConnector(),
                                sim.StaticSynapse(weight=inhibitory_weight, delay=TIMESTEP),
                                label='inh_output to output',
                                receptor_type='inhibitory')

            return prjs


    def _get_recorded(self, layer):
        data = {}
        if layer == 'input':
            pops = self.input_populations()
            for i in pops:
                data[i] = grab_data(pops[i])

        elif layer == 'gabor':
            _, pops = self.gabor_populations()
            for i in pops:
                idict = data.get(i, {})
                for r in pops[i]:
                    rdict = idict.get(r, {})
                    for c in pops[i][r]:
                        rdict[c] = grab_data(pops[i][r][c])
                    idict[r] = rdict
                data[i] = idict
        else:
            data[0] = grab_data(self._network['populations'][layer])

        return data

    def run_pynn(self):
        net = self._network
        # pprint(net)

        logging.info("\tRunning experiment for {} milliseconds".format(net['run_time']))
        sim.run(net['run_time'])

        records = {}
        for pop in net['populations']:
            if pop in record_spikes:
                records[pop] = self._get_recorded(pop)

        weights = {}
        for proj in net['projections']:
            if proj in record_weights:
                weights[proj] = grab_weights(net['projections'][proj])


        sim.end()


        ### todo: change start and end for labels and runtimes
        dt = self.params['sim']['sample_dt']
        nclass = self.params['sim']['num_classes']
        ntrain = self.params['sim']['samples_per_class']
        start_t = dt * nclass * ntrain
        cls_labels = self.in_labels[int(start_t//dt):]
        spk_p_class = spiking_per_class(cls_labels,
                                        records['output'][0]['spikes'],
                                        start_t, net['run_time'], dt),

        data = {
            'recs': records,
            'weights': weights,
            'input': {
                'labels': self.in_labels,
                'spikes': self.inputs,
                'shapes': self.in_shapes,
            },
            'params': self.params,
            'gabor': {
                'shapes': self.gabor_shapes,
            },
            'analysis':{
                'per_class': spk_p_class
            }
        }



        return data


def grab_data(pop):
    data = pop.get_data()
    try:
        spikes = spikes_from_data(data)
    except:
        spikes = []
    try:
        voltage = voltage_from_data(data)
    except:
        voltage = []
    return {'spikes': spikes, 'voltage': voltage}


def spikes_from_data(data):

    spikes = [[] for _ in range(len(data.segments[0].spiketrains))]
    for train in data.segments[0].spiketrains:
        spikes[int(train.annotations['source_index'])][:] = \
            [float(t) for t in train]
    return spikes


def voltage_from_data(data):
    volts = data.filter(name='v')
    return [[[float(a), float(b), float(c)] for a, b, c in volts]]

def safe_get_weights(p):
    try:
        return p.getWeights(format='array')
    except:
        return []

def grab_weights(proj):
    if isinstance(proj, dict): #gabor connections are a lot! :O
        w = {}
        for k in proj:
            wk = {}
            if isinstance(proj[k], dict):
                for r in proj[k]:
                    wr = {}
                    if isinstance(proj[k][r], dict):
                        for c in proj[k][r]:
                            wc = {}
                            if isinstance(proj[k][r][c], dict):
                                for x in proj[k][r][c]:
                                    # print(k,r,c,x, proj[k][r][c][x])
                                    wc[x] = safe_get_weights(proj[k][r][c][x])
                            else:
                                # print(k, r, c, proj[k][r][c])
                                wc[-1] = safe_get_weights(proj[k][r][c])
                            wr[c] = wc
                    else:
                        # print(k, r, proj[k][r])
                        wr[-1] = safe_get_weights(proj[k][r])
                    wk[r] = wr
            else:
                # print(k, proj[k])
                wk[-1] = safe_get_weights(proj[k])

            w[k] = wk

        return w
    else:
        try:
            return safe_get_weights(proj)
        except:
            return []