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
    
    def setup(self, timestep=1.0, min_delay=1.0, **kwargs):
        setup_args = {'timestep': timestep, 'min_delay': min_delay}
        self._extra_config = kwargs
        
        if self.sim_name == BSS: #do extra setup for BrainScaleS
            wafer = kwargs['wafer']
            
            self.init_logging()
            
            marocco, runtime = self._init_marocco(wafer)
            self._marocco = marocco
            self._runtime = runtime
            
            setup_args['marocco'] = marocco
            setup_args['marocco_runtime'] = runtime
        
        self._setup_args = setup_args
        self.sim.setup(**setup_args)

    def init_logging(self):
        init_logger("WARN", [
                    ("guidebook", "DEBUG"),
                    ("marocco", "DEBUG"),
                    ("Calibtic", "DEBUG"),
                    ("sthal", "INFO")
                ])
        self._log = pylogging.get("guidebook")
    
    def _init_marocco(self, wafer):
        marocco = PyMarocco()
        marocco.default_wafer = C.Wafer(wafer)
        runtime = Runtime(marocco.default_wafer)

        return marocco, runtime
        
    
    def do_placement(self, pop, coord):
        if not hasattr(self, '_hicann'):
            self._hicann = {}
            
        self._hicann[coord] = C.HICANNOnWafer(C.Enum(coord))
        marocco.manual_placement.on_hicann(pop, self._hicann[coord])
        
    
    

    def _set_sthal_params(wafer, gmax, gmax_div):
        # change low-level parameters before configuring hardware
        """
        synaptic strength:
        gmax: 0 - 1023, strongest: 1023
        gmax_div: 1 - 15, strongest: 1
        """
        assert gmax >= 0 and gmax <= 1023, "gmax has to be in the range [0, 1023]"
        assert gmax_div >= 1 and gmax_div <= 15, "gmax_div has to be in the range [1, 15]"
        

        # for all HICANNs in use
        for hicann in wafer.getAllocatedHicannCoordinates():

            fgs = wafer[hicann].floating_gates

            # set parameters influencing the synaptic strength
            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

            for driver in C.iter_all(C.SynapseDriverOnHICANN):
                for row in C.iter_all(C.RowOnSynapseDriver):
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.left, gmax_div)
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.right, gmax_div)

            # don't change values below
            for ii in xrange(fgs.getNoProgrammingPasses()):
                cfg = fgs.getFGConfig(C.Enum(ii))
                cfg.fg_biasn = 0
                cfg.fg_bias = 0
                fgs.setFGConfig(C.Enum(ii), cfg)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
                fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)
    
    
    def run(self, duration, gmax=1023, gmax_div=1):
        if self.sim_name == BSS:
            if self._first_run:
                marocco.skip_mapping = False
                marocco.backend = PyMarocco.None

                self.sim.reset()
                self.sim.run(duration)
                
                self._set_sthal_params(self.runtime.wafer(), gmax, gmax_div)
                
                marocco.skip_mapping = True
                marocco.backend = PyMarocco.Hardware
                
                # Full configuration during first step
                marocco.hicann_configurator = PyMarocco.ParallelHICANNv4Configurator
                self._first_run = False
            else:
                self.sim.run(duration)
        else:
            self.sim.run(duration)
    
    def reset(self, skip_marocco_checks=True):
        self.sim.reset()
        if self.sim_name == BSS and not self._first_run:
            # only change digital parameters from now on
            self._marocco.hicann_configurator = PyMarocco.NoResetNoFGConfigurator
            # skip checks
            if skip_marocco_checks:
                marocco.verification = PyMarocco.Skip
                marocco.checkl1locking = PyMarocco.SkipCheck

    @property
    def sim(self):
        return self._sim
    
    @property
    def sim_name(self):
        return self._sim_name

    
    def _is_v9(self):
        return ('genn' in self.sim.__name__)

    def _ver(self):
        return (9 if self._is_v9() else 7)


    def Pop(self, size, cell_class, params, label=None):
        sim = self.sim
        
        if self._ver() == 7:
            return sim.Population(size, cell_class, params, label=label)
        else:
            return sim.Population(size, cell_class(**params), label=label)


    def Proj(self, source_pop, dest_pop, conn_class, weights, delays, 
             target='excitatory', stdp=None, label=None, conn_params={}):
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
        sim = self.sim
        if self._ver() == 7:
            data = np.array(pop.getSpikes())
            ids = np.unique(data[:, 0])
            return [data[np.where(data[:, 0] == nid)][:, 1] if nid in ids else []
                    for nid in range(pop.size)]
        else:
            data = pop.get_data().segments[segment]
            return np.array(data.spiketrains)

    def get_weights(self, proj, format='array'):
        return np.array(proj.getWeights(format=format))


    def set_pop_attr(self, pop, attr_name, attr_val):
        if self._ver() == 7:
            pop.set(attr_name, attr_val)
        else:
            pop.set(**{attr_name: attr_val})



