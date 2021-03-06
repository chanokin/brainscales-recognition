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
from .partitioning import SplitPop, SplitPopulation, \
                          SplitArrayPopulation, SplitProjection

import os
try:
    from pyhalbe import HICANN
    import pyhalbe.Coordinate as C
    from pysthal.command_line_util import init_logger
    from pymarocco import PyMarocco, Defects
    from pymarocco.runtime import Runtime
    from pymarocco.coordinates import LogicalNeuron
    from pymarocco.results import Marocco
    from pymarocco import Defects
    from pysthal.command_line_util import init_logger

    init_logger("WARN", [
        ("guidebook", "INFO"),
        ("marocco", "INFO"),
        ("Calibtic", "INFO"),
        ("sthal", "INFO")
    ])

except:
    pass


class PyNNAL(object):
    """
    A PyNN Abstraction Layer (yet another?) used to reduce the times
    we may need to adjust our scripts to run in different versions of PyNN.
    Required mainly due to different PyNN implementations lagging or moving ahead.
    """
    def __init__(self, simulator, max_subpop_size=np.inf):
        if isinstance(simulator, str) or type(simulator) == type(u''):
            simulator = backend_setup(simulator)

        self._sim = simulator
        sim_name = simulator.__name__
        self._max_subpop_size = max_subpop_size
        self._wafer = None
        if GENN in sim_name:
            self._sim_name = GENN
        elif NEST in sim_name:
            self._sim_name = NEST
        elif BSS_BACK in sim_name:
            self._max_subpop_size = BSS_MAX_SUBPOP_SIZE
            self._sim_name = BSS
            self.marocco = None
        else:
            raise Exception("Not supported simulator ({})".format(sim_name))

        self._first_run = True
        self._graph = Graph()

    def __del__(self):
        try:
            self.end()
        except:
            pass

    def NumpyRNG(self, seed=None):
        try:
            rng = self._sim.NumpyRNG(seed=seed)
        except Exception as inst:
            rng = self._sim.random.NumpyRNG(seed=seed)
        # finally:
        #     raise Exception("Can't find the NumpyRNG class!")

        return rng

    @property
    def sim(self):
        return self._sim

    @sim.setter
    def sim(self, v):
        self._sim = v

    @property
    def sim_name(self):
        return self._sim_name

    @sim_name.setter
    def sim_name(self, v):
        self._sim_name = v

    def setup(self, timestep=1.0, min_delay=1.0, per_sim_params={}, **kwargs):
        setup_args = {'timestep': timestep, 'min_delay': min_delay}
        self._extra_config = per_sim_params
        
        if self.sim_name == BSS: #do extra setup for BrainScaleS
            wafer = per_sim_params.get("wafer", None)
            marocco = per_sim_params.get("marocco", PyMarocco())
            marocco.backend = PyMarocco.Hardware
            if wafer is not None:
                per_sim_params.pop('wafer')
                marocco.default_wafer = C.Wafer(wafer)
                runtime = Runtime(marocco.default_wafer)
                setup_args['marocco_runtime'] = runtime
                self.runtime = runtime
                self._wafer = WAL(wafer_id=wafer)
                
            calib_path = per_sim_params.get("calib_path",
                            "/wang/data/calibration/brainscales/WIP-2018-09-18")
            
            marocco.calib_path = calib_path
            marocco.defects.path = marocco.calib_path
            marocco.verification = PyMarocco.Skip
            marocco.checkl1locking = PyMarocco.SkipCheck
            marocco.continue_despite_synapse_loss = True
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
                # self._do_BSS_placement()
                # self.marocco.skip_mapping = True
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
                '''REMOVE THIS!!! just for testing!!!'''
                # self._wafer = WAL(wafer_id=33) #TODO: REMEMBER TO DELETE THIS!!!!
                # self._do_BSS_placement() #TODO: REMEMBER TO DELETE THIS!!!!

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
        return ('genn' in self._sim.__name__)

    def _ver(self):
        return (9 if self._is_v9() else 7)

    def _get_obj(self, obj_name):
        return getattr(self._sim, obj_name)
    
    def Pop(self, size, cell_class, params, label=None, shape=None,
        max_sub_size=None, hicann_id=None):

        if max_sub_size is None:
            max_sub_size = self._max_subpop_size
        if is_string(cell_class): #convert from text representation to object
            txt_class = cell_class
            cell_class = self._get_obj(cell_class)
        else:
            txt_class = cell_class.__name__

        is_source_pop = txt_class.startswith('SpikeSource')
        
        sim = self.sim
        if self._sim_name == BSS and txt_class.lower() == 'spikesourcearray' and \
           params['spike_times'] and isinstance(params['spike_times'][0], list):
            spop = SplitArrayPopulation(self, size, cell_class, params, label, 
                                        shape, max_sub_size=1)
            for pop_dict in spop._populations:
                pop = pop_dict['pop']
                if hicann_id is not None:
                    hicann = C.HICANNOnWafer(C.Enum(hicann_id))
                    self.marocco.manual_placement.on_hicann(pop, hicann)

            self._graph.add(pop, is_source_pop)
            if self._graph.width < 1:
                self._graph.width = 1

            return spop
            
        elif size <= max_sub_size or is_source_pop:
            if self._ver() == 7:
                pop = sim.Population(size, cell_class, params, label=label)
                if self._sim_name == BSS and hicann_id is not None:
                    hicann = C.HICANNOnWafer(C.Enum(hicann_id))
                    self.marocco.manual_placement.on_hicann(pop, hicann)
            else:
                pop = sim.Population(size, cell_class(**params), label=label)

            self._graph.add(pop, is_source_pop)
            if self._graph.width < 1:
                self._graph.width = 1



            return pop
        else:
            width = calc_n_part(size, max_sub_size)
            if self._graph.width < width:
                self._graph.width = width
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SplitPopulation(self, size, cell_class, params, label, shape,
                    max_sub_size)

    def _get_stdp_dep(self, config):
        _dep = self._get_obj(config['name'])
        return _dep(**config['params'])



    def parse_conn_params(self, param):
        if isinstance(param, dict):
            dist = param['type']
            dist_params = param['params']
            rng = self._sim.NumpyRNG(param['seed'])
            return self._sim.RandomDistribution(dist, dist_params, rng)
        else:
            return param


    def Proj(self, source_pop, dest_pop, conn_class, weights, delays=1,
             target='excitatory', stdp=None, label=None, conn_params={}):

        if isinstance(source_pop, SplitPop) or \
            isinstance(dest_pop, SplitPop):
            ### first argument is this PYNNAL instance, needed to loop back here!
            ### a bit spaghetti but it's less code :p
            return SplitProjection(self, source_pop, dest_pop, conn_class, weights, delays,
             target=target, stdp=stdp, label=label, conn_params=conn_params)
            
        if is_string(conn_class): #convert from text representation to object
            conn_text = conn_class
            conn_class = self._get_obj(conn_class)
        else:
            conn_text = conn_class.__name__

        sim = self._sim

        weights = self.parse_conn_params(weights)
        delays = self.parse_conn_params(delays)


        if self._ver() == 7:
            """ Extract output population from NestImagePopulation """
            pre_pop = source_pop.out if isinstance(source_pop, NestImagePopulation)\
                        else source_pop

            if conn_text.startswith('From'):
                tmp = conn_params.copy() 
            else:
                tmp = conn_params.copy()
                tmp['weights'] = weights
                tmp['delays'] = delays
                
            conn = conn_class(**tmp)
            
            if stdp is not None:
                rule_txt = stdp.get('rule', 'STDPMechanism')
                rule = self._get_obj(rule_txt)
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
                            slow=rule(
                                timing_dependence=self._get_stdp_dep(stdp['timing_dependence']),
                                weight_dependence=self._get_stdp_dep(stdp['weight_dependence']))
                            )
            else:
                syn_dyn = None

            proj = sim.Projection(pre_pop, dest_pop, conn,
                    target=target, synapse_dynamics=syn_dyn, label=label)
            
        else:
            if stdp is not None:
                rule_txt = stdp.get('rule', 'STDPMechanism')
                rule = self._get_obj(rule_txt)
                ### Compatibility between versions - change parameters to the other description
                if 'A_plus' in stdp['weight_dependence']['params']:
                    stdp['timing_dependence']['params']['A_plus'] = \
                        stdp['weight_dependence']['params']['A_plus']
                    del stdp['weight_dependence']['params']['A_plus']

                if 'A_minus' in stdp['weight_dependence']['params']:
                    stdp['timing_dependence']['params']['A_minus'] = \
                        stdp['weight_dependence']['params']['A_minus']
                    del stdp['weight_dependence']['params']['A_minus']

                synapse = rule(
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
                data = pop.get_data()
                segments = data.segments
                spiketrains = segments[0].spiketrains
                spikes[:] = [[] for _ in range(pop.size)]
                for train in spiketrains:
                    ### NOTE: had to remove units because pyro don't like numpy!
                    spikes[int(train.annotations['source_index'])][:] = \
                                                        [float(t) for t in train]
        
        return spikes


    def get_weights(self, proj, format='array'):
        ### NOTE: screw the non-array representation!!! Who thought that was a good idea?

        ###
        # if self._sim_name == GENN:
        #     ### TODO: we want to return arrays here! Currently, it's just not possible.
        #     print("\n\nTrying to get weights for GeNN\n\n")
        #     weights = proj.get('weight', format='list', with_address=False)
        #     print("\n\nAFTER --- Trying to get weights for GeNN\n\n")
        #     return np.array(weights)

        format = 'array'
        return np.array(proj.getWeights(format=format))


    def set_pop_attr(self, pop, attr_name, attr_val):
        if self._ver() == 7:
            pop.set(attr_name, attr_val)
        else:
            pop.set(**{attr_name: attr_val})

    def check_rec(self, recording):
        if recording not in ['spikes', 'v', 'gsyn', 'dvdt']:
            raise Exception('Recording {} is not supported'.format(recording))


    def set_recording(self, pop, recording):
        self.check_rec(recording)
        pop_name = pop.label.replace(' ', '_')
        if self._ver() == 7:
            if recording == 'spikes':
                pop.record()
            else:
                rec = getattr(pop, 'record_'+recording) #indirectly get method
                rec() #execute method :ugly-a-f:
        else:

            pop.record(recording)#, to_file='pop_%s_recording_%s.npz'%(pop_name, recording))

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
        placer = WaferPlacer(self._graph, self._wafer)
        placer._place()
        self._graph.update_places(placer.places)
