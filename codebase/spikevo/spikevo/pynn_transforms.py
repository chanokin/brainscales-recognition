from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np
import numbers
from . import *
from .image_input import NestImagePopulation

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
        max_sub_size=MAX_SUBPOP_SIZE):
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
        self.subpops = pops



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
        if label is None:
            self.label = "SplitProjection from {} to {}".format(
                            self.source.label, self.destination.label)
        else:
            self.label = label
        
        partition()

    def partition(self):
        pynnx = self.pynnal
        src = self.source
        dst = self.destination
        
        if isinstance(src, SplitPopulation):
            for src_part in src:
                pre_ids, pre = part['ids'], part['pop']
                
                if isinstance(dst, SplitPopulation):
                    for dst_part in src:
                        post_ids, post = part['ids'], part['pop']
                else:
                    
    def _proj(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        cname = conn if type(conn) == type(u'') else conn.__name__
        if cname=='FromListConnector':
            from_list_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        else:
            stats_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)

    def stats_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pass

    def from_list_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pass



class PyNNAL(object):
    """
    A PyNN Abstraction Layer (yet another?) used to reduce the times
    we may need to adjust our scripts to run in different versions of PyNN.
    Required mainly due to different PyNN implementations lagging or moving ahead.
    """
    def __init__(self, simulator):
        
        self._sim = simulator
        sim_name = simulator.__name__
        if GENN in sim_name:
            self._sim_name = GENN
        elif NEST in sim_name:
            self._sim_name = NEST
        elif BSS_BACK in sim_name:
            self._sim_name = BSS
        else:
            raise Exception("Not supported simulator ({})".format(sim_name))

        self._first_run = True

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
            marocco = PyMarocco()
            marocco.backend = PyMarocco.Hardware
            marocco.calib_path = per_sim_params.get('calib_path',
                            "/wang/data/calibration/brainscales/WIP-2018-09-18")
            
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
                # self.marocco.skip_mapping = False
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

    def get_obj(self, obj_name):
        return getattr(self._sim, obj_name)
    
    def Pop(self, size, cell_class, params, label=None, shape=None,
        max_sub_size=MAX_SUBPOP_SIZE):
        if type(cell_class) == type(u''): #convert from text representation to object
            cell_class = self.get_obj(cell_class)

        sim = self.sim
        
        if size <= max_sub_size:
            if self._ver() == 7:
                return sim.Population(size, cell_class, params, label=label)
            else:
                return sim.Population(size, cell_class(**params), label=label)
        else:
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SubPopulation(self, size, cell_class, params, label, shape, 
                    max_sub_size)



    def Proj(self, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={}):

        if isinstance(source_pop, SplitPopulation) or \
            isinstance(source_pop, SplitPopulation):
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SubProjection(self, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={})
            
        if type(conn_class) == type(u''): #convert from text representation to object
            conn_class = self.get_obj(conn_class)

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
                syn_dyn = sim.SynapseDynamics(
                            slow=sim.STDPMechanism(
                                timing_dependence=stdp['timing_dependence'],
                                weight_dependence=stdp['weight_dependence'])
                            )
            else:
                syn_dyn = None

            return sim.Projection(pre_pop, dest_pop, conn,
                    target=target, synapse_dynamics=syn_dyn, label=label)
            
        else:
            if stdp is not None:
                synapse = sim.STDPMechanism(
                    timing_dependence=stdp['timing_dependence'],
                    weight_dependence=stdp['weight_dependence'],
                    weight=weights, delay=delays)
            else:
                synapse = sim.StaticSynapse(weight=weights, delay=delays)

            return sim.Projection(source_pop, dest_pop, conn_class(**conn_params), 
                    synapse, receptor_type=target, label=label)


    def get_spikes(self, pop, segment=0):
        if self._ver() == 7:
            data = np.array(pop.getSpikes())
            ids = np.unique(data[:, 0])
            return [data[np.where(data[:, 0] == nid)][:, 1].tolist() \
                        if nid in ids else [] for nid in range(pop.size)]
        else:
            spiketrains = pop.get_data().segments[0].spiketrains
            spikes = [[] for _ in spiketrains]
            for train in spiketrains:
                ### NOTE: had to remove units because pyro or pypet don't like it!
                spikes[int(train.annotations['source_index'])][:] = \
                    [float(t) for t in train] 
            return spikes

    def get_weights(self, proj, format='array'):
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
            pop.get_data().segments[0].filter(name=recording)
        

