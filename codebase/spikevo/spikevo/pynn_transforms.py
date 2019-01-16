from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import numbers
from . import *
from .image_input import NestImagePopulation
from .wafer import Wafer as WAL
from .graph import Graph, Node
from .brainscales_placement import *

import os
try:
    from pyhalbe import HICANN
    import pyhalbe.Coordinate as C
    from pysthal.command_line_util import init_logger
    from pymarocco import PyMarocco, Defects
    from pymarocco.runtime import Runtime
    from pymarocco.coordinates import LogicalNeuron
    from pymarocco.results import Marocco
except:
    pass


class SplitPopulation(object):
    """
        When using the BrainScaleS toolchain we faced some problems with the
        partition and place-and-route algorithms. The initial problem is 
        that the tools break when using certain connectivity (e.g. populations
        greater than 200 with one-to-one connectors). This class attempts to
        avoid the problem by partitionining before the toolchain requires it.
    """
    ### This seems like reinventing the wheel and I shouldn't have to!
    def __init__(self, pynnal, size, cell_class, params, label=None, shape=None,
        max_sub_size=BSS_MAX_SUBPOP_SIZE):
        """
            pynnal: PyNN Abstraction Layer instance, we use it to avoid re-coding
                different cases for different PyNN versions while creating
                the populations.
            size: original size of the population
            cell_class: original PyNN cell type of the population
            params: original PyNN parameters for the given cell type
            label: original population label/name
            shape: shape of the original population (currently only 1D supported)
            max_sub_size: size of the sub-populations to be created
        """
        self.pynnal = pynnal
        self.size = size
        self.cell_class = cell_class
        self.params = params
        if label is None:
            self.label = "SplitPopulation ({:05d})".format(
                                            np.random.randint(0, 99999))
        else:
            self.label = label

        if shape is None:
            self.shape = (size,) # tuple expressing grid size, 1D by default
        else:
            assert np.prod(shape) == size, \
                "Total number of neurons should equal grid dimensions"
            self.shape = shape

        ### TODO: this will likely change if shape is not 1D
        self.max_sub_size = max_sub_size
        self.n_sub_pops = calc_n_part(self.size, self.max_sub_size) 
        
        ### actually do the partitioning
        self.partition()

    def partition(self):
        if len(self.shape) == 1:
            pops = []
            count = 0
            for i in range(self.n_sub_pops):
                size = min(self.max_sub_size, self.size - count)
                ids = np.range(count, count+size)
                count += self.max_sub_size
                label = self.label + " - sub %d"%(i+1)
                pops.append({
                    'ids': ids,
                    'pop': self.pynnal.Pop(size, self.cell_class, label)
                })

        ### TODO: deal with 2D, 3D!
        self._populations = pops



class SplitProjection(object):
    """
    Since we had to pre-partition the populations, now we need to split-up
    the projections as well.
    """
    ### This seems like reinventing the wheel and I shouldn't have to!
    def __init__(self, pynnal, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={}):
        self.pynnal = pynnal
        self.source = source_pop
        self.destination = dest_pop
        self.conn_class = conn_class
        self.weights = weights,
        self.delays = delays,
        self.target = target
        self.stdp = stdp
        self.conn_params = conn_params
        self._projections = None
        if label is None:
            self.label = 'SplitProjection from {} to {}'.format(
                            self.source.label, self.destination.label)
        else:
            self.label = label
        
        self.partition()

    def partition(self):
        src = self.source
        dst = self.destination
        conn, params = self.conn_class, self.conn_params
        w, d, tgt = self.weights, self.delays, self.target
        stdp = self.stdp
        
        if isinstance(src, SplitPopulation):
            pres = src._populations
        else:
            pres = [{'ids': np.arange(src.size), 'pop': src}]
        
        if isinstance(dst, SplitPopulation):
            posts = dst._populations
        else:
            posts = [{'ids': np.arange(dst.size), 'pop': dst}]
        
        projs = {}
        for src_part in src:
            pre_ids, pre = src_part['ids'], src_part['pop']
            src_prjs = projs.get(pre.label, dict)
            
            for dst_part in dst:
                post_ids, post = dst_part['ids'], dst_part['pop']
                lbl = '{} sub {} - {}'.format(self.label, pre.label, post.label)
                    
                proj = self._proj(src_part, dst_part, conn, w, d, 
                        tgt, params, lbl, stdp)

                if proj is None:
                    continue
                
                src_prjs[post.label] = {'ids': {'pre': pre_ids, 'post': post_ids},
                                        'proj': proj}

            projs[pre.label] = src_prjs

        self._projections = projs


    def _proj(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        cname = conn if type(conn) == type(u'') else conn.__name__
        if cname.startswith('FromList'):
            return self.from_list_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('OneToOne'):
            return self.one_to_one_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('AllToAll'):
            return self.all_to_all_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('FixedProbability'):
            return self.stats_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        else:
            raise Exception("unsupported connection for splitting")
            # return None


    def stats_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal
        return pynnal.Proj(pre['pop'], post['pop'], conn, w, d,
                target=tgt, stdp=stdp, label=lbl, conn_params=params)

    def one_to_one_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal
        if pre['ids'][0] != post['ids'][0] or pre['ids'][-1] != post['ids'][-1]:
            return None

        return pynnal.Proj(pre['pop'], post['pop'], conn, w, d, 
                target=tgt, stdp=stdp, label=lbl, conn_params=params)

    def all_to_all_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        return self.stats_connector(pre, post, conn, w, d, tgt, params, lbl, stdp)

    def from_list_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal
        clist = params['conn_list']
        if isinstance(clist, list):
            clist = np.array(clist)

        whr = np.where(np.intersect1d(
                np.intersect1d(clist[:, 0], pre['ids'])[0],
                np.intersect1d(clist[:, 1], post['ids'])[0]))[0]
        cp = {'conn_list': clist[whr,:]}

        return pynnal.Proj(pre['pop'], post['pop'], conn, w[whr], d[whr], 
                target=tgt, stdp=stdp, label=lbl, conn_params=cp)

    def getWeights(self, format='array'):
        pynnal = self.pynnal
        mtx = np.ones((self.source.size, self.destination.size)) * np.inf
        for row in self._projections:
            for part in row:
                pre_ids = part['ids']['pre']
                r0, rN = pre_ids[0], pre_ids[-1]
                post_ids = part['ids']['post']
                c0, cN = post_ids[0], post_ids[-1]
                weights = pynnal.getWeights(part['proj'], format=format)
                mtx[r0:rN, c0:cN] = weights
        return mtx


class PyNNAL(object):
    """
    A PyNN Abstraction Layer (yet another?) used to reduce the times
    we may need to adjust our scripts to run in different versions of PyNN.
    Required mainly due to different PyNN implementations lagging or moving ahead.
    """
    def __init__(self, simulator):
        if isinstance(simulator, str) or type(simulator) == type(u''):
            simulator = backend_setup(simulator)

        self._sim = simulator
        sim_name = simulator.__name__
        self._max_subpop_size = np.inf
        self._wafer = None
        if GENN in sim_name:
            self._sim_name = GENN
        elif NEST in sim_name:
            self._sim_name = NEST
        elif BSS_BACK in sim_name:
            self._max_subpop_size = BSS_MAX_SUBPOP_SIZE
            self._sim_name = BSS
        else:
            raise Exception("Not supported simulator ({})".format(sim_name))

        self._first_run = True
        self._graph = Graph()

    def __del__(self):
        try:
            self.end()
        except:
            pass

    @property
    def sim(self):
        return self._sim
    
    @property
    def sim_name(self):
        return self._sim_name


    def setup(self, timestep=1.0, min_delay=1.0, per_sim_params={}, **kwargs):
        setup_args = {'timestep': timestep, 'min_delay': min_delay}
        self._extra_config = per_sim_params
        
        if self.sim_name == BSS: #do extra setup for BrainScaleS
            wafer = per_sim_params.get('wafer', None)
            marocco = PyMarocco()
            marocco.backend = PyMarocco.Hardware
            if wafer is not None:
                per_sim_params.pop('wafer')
                marocco.default_wafer = C.Wafer(wafer)
                runtime = Runtime(marocco.default_wafer)
                setup_args['marocco_runtime'] = runtime
                self.runtime = runtime
                self._wafer = WAL(wafer_id=wafer)



            marocco.calib_path = per_sim_params.get('calib_path',
                            '/wang/data/calibration/brainscales/WIP-2018-09-18')
            
            marocco.defects.path = marocco.calib_path
            marocco.verification = PyMarocco.Skip
            marocco.checkl1locking = PyMarocco.SkipCheck

            per_sim_params.pop('calib_path', None)
            
            setup_args['marocco'] = marocco
            self.marocco = marocco



        for k in per_sim_params:
            setup_args[k] = per_sim_params[k]

        self._setup_args = setup_args
        
        self._sim.setup(**setup_args)

    def run(self, duration, gmax=1023, gmax_div=1):
        if self.sim_name == BSS:
            if self._first_run:
                self._do_BSS_placement()
                self.marocco.skip_mapping = True
                # self.marocco.backend = PyMarocco.None

                # self.sim.reset()
                self._sim.run(duration)
                
                # self._set_sthal_params(self.runtime.wafer(), gmax, gmax_div)
                
                # marocco.skip_mapping = True
                # marocco.backend = PyMarocco.Hardware
                
                # # Full configuration during first step
                # marocco.hicann_configurator = PyMarocco.ParallelHICANNv4Configurator
                self._first_run = False
            else:
                self._sim.run(duration)
        else:
            if self._first_run:
                self._first_run = False
            self._sim.run(duration)
    
    def reset(self, skip_marocco_checks=True):
        self._sim.reset()
        if self.sim_name == BSS and not self._first_run:
            # only change digital parameters from now on
            self.marocco.hicann_configurator = PyMarocco.NoResetNoFGConfigurator
            # skip checks
            if skip_marocco_checks:
                self.marocco.verification = PyMarocco.Skip
                self.marocco.checkl1locking = PyMarocco.SkipCheck
    
    def end(self):
        self._sim.end()
    
    def _is_v9(self):
        return ('genn' in self.sim.__name__)

    def _ver(self):
        return (9 if self._is_v9() else 7)

    def _get_obj(self, obj_name):
        return getattr(self._sim, obj_name)
    
    def Pop(self, size, cell_class, params, label=None, shape=None,
        max_sub_size=None):

        if max_sub_size is None:
            max_sub_size = self._max_subpop_size
        if type(cell_class) == type(u''): #convert from text representation to object
            txt_class = cell_class
            cell_class = self._get_obj(cell_class)
        else:
            txt_class = cell_class.__name__
            # cname = cell_class
        # else:
            # cname = cell_class.__name__
        # is_source_pop = cname.startswith('SpikeSource')
        
        sim = self.sim
        
        if size <= max_sub_size:# or is_source_pop:
            if self._ver() == 7:
                pop = sim.Population(size, cell_class, params, label=label)
            else:
                pop = sim.Population(size, cell_class(**params), label=label)

            is_source = 'SpikeSource' in txt_class
            self._graph.add(pop, is_source)

            return pop
        else:
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SplitPopulation(self, size, cell_class, params, label, shape,
                    max_sub_size)

    def _get_stdp_dep(self, config):
        _dep = self._get_obj(config['name'])
        return _dep(**config['params'])



    def Proj(self, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={}):

        if isinstance(source_pop, SplitPopulation) or \
            isinstance(source_pop, SplitPopulation):
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SplitProjection(self, source_pop, dest_pop, conn_class, weights, delays,
             target=target, stdp=None, label=None, conn_params={})
            
        if type(conn_class) == type(u''): #convert from text representation to object
            conn_class = self._get_obj(conn_class)

        sim = self.sim
            
        if self._ver() == 7:
            """ Extract output population from NestImagePopulation """
            pre_pop = source_pop.out if isinstance(source_pop, NestImagePopulation)\
                        else source_pop

            tmp = conn_params.copy()
            tmp['weights'] = weights
            tmp['delays'] = delays
            conn = conn_class(**tmp)
            
            if stdp is not None:
                ### Compatibility between versions - change parameters to the other description
                if 'A_plus' in stdp['timing_dependence']['params']:
                    stdp['weight_dependence']['params']['A_plus'] = \
                        stdp['timing_dependence']['params']['A_plus']
                    del stdp['timing_dependence']['params']['A_plus']

                if 'A_minus' in stdp['timing_dependence']['params']:
                    stdp['weight_dependence']['params']['A_minus'] = \
                        stdp['timing_dependence']['params']['A_minus']
                    del stdp['timing_dependence']['params']['A_minus']

                syn_dyn = sim.SynapseDynamics(
                            slow=sim.STDPMechanism(
                                timing_dependence=self._get_stdp_dep(stdp['timing_dependence']),
                                weight_dependence=self._get_stdp_dep(stdp['weight_dependence']))
                            )
            else:
                syn_dyn = None

            proj = sim.Projection(pre_pop, dest_pop, conn,
                    target=target, synapse_dynamics=syn_dyn, label=label)
            
        else:
            if stdp is not None:
                ### Compatibility between versions - change parameters to the other description
                if 'A_plus' in stdp['weight_dependence']['params']:
                    stdp['timing_dependence']['params']['A_plus'] = \
                        stdp['weight_dependence']['params']['A_plus']
                    del stdp['weight_dependence']['params']['A_plus']

                if 'A_minus' in stdp['weight_dependence']['params']:
                    stdp['timing_dependence']['params']['A_minus'] = \
                        stdp['weight_dependence']['params']['A_minus']
                    del stdp['weight_dependence']['params']['A_minus']

                synapse = sim.STDPMechanism(
                    timing_dependence=self._get_stdp_dep(stdp['timing_dependence']),
                    weight_dependence=self._get_stdp_dep(stdp['weight_dependence']),
                    weight=weights, delay=delays)
            else:
                synapse = sim.StaticSynapse(weight=weights, delay=delays)

            proj = sim.Projection(source_pop, dest_pop, conn_class(**conn_params),
                    synapse_type=synapse, receptor_type=target, label=label)


        self._graph.plug(source_pop, dest_pop)
        return proj

    def get_spikes(self, pop, segment=0):
        spikes = []
        if isinstance(pop, SplitPopulation):
            ### TODO: deal with 2D/3D pops
            for part in pop._populations:
                spikes += self.get_spikes(part['pop'])
        else:
            if self._ver() == 7:
                data = np.array(pop.getSpikes())
                ids = np.unique(data[:, 0])
                spikes[:] = [data[np.where(data[:, 0] == nid)][:, 1].tolist() \
                                if nid in ids else [] for nid in range(pop.size)]
            else:
                spiketrains = pop.get_data().segments[0].spiketrains
                spikes[:] = [[] for _ in spiketrains]
                for train in spiketrains:
                    ### NOTE: had to remove units because pyro or pypet don't like it!
                    spikes[int(train.annotations['source_index'])][:] = \
                        [float(t) for t in train] 
        
        return spikes


    def get_weights(self, proj, format='array'):
        ### NOTE: screw the non-array representation!!! Who thought that was a good idea?
        format = 'array'
        return np.array(proj.getWeights(format=format))


    def set_pop_attr(self, pop, attr_name, attr_val):
        if self._ver() == 7:
            pop.set(attr_name, attr_val)
        else:
            pop.set(**{attr_name: attr_val})

    def check_rec(self, recording):
        if recording not in ['spikes', 'v', 'gsyn']:
            raise Exception('Recording {} is not supported'.format(recording))


    def set_recording(self, pop, recording):
        self.check_rec(recording)
        if self._ver() == 7:
            if recording == 'spikes':
                pop.record()
            else:
                rec = getattr(pop, 'record_'+recording) #indirectly get method
                rec() #execute method :ugly-a-f:
        else:             
            pop.record(recording)

    def get_record(self, pop, recording):
        self.check_rec(recording)
        if recording == 'spikes':
            return self.get_spikes(pop)
        elif self._ver() == 7:
            record = getattr(pop, 'get_'+recording) #indirectly get method
            return record() #execute method :ugly-a-f:
        else:
            return pop.get_data().segments[0].filter(name=recording)


    def _do_BSS_placement(self):
        self.placer = WaferPlacer(self._graph, self._wafer)
        self.placer._place()