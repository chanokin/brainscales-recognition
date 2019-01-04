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

class PyNNAL(object):
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
    
    def Pop(self, size, cell_class, params, label=None):
        if type(cell_class) == type(u''): #convert from text representation to object
            cell_class = self.get_obj(cell_class)

        sim = self.sim
        
        if self._ver() == 7:
            return sim.Population(size, cell_class, params, label=label)
        else:
            return sim.Population(size, cell_class(**params), label=label)


    def Proj(self, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={}):
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
        

