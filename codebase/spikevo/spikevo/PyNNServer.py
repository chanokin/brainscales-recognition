from __future__ import (print_function,
                        unicode_literals,
                        division)
import Pyro4
from spikevo.pynn_transforms import PyNNAL
from spikevo import *

@Pyro4.expose
# @Pyro4.behavior(instance_mode="percall") #single instance per call
@Pyro4.behavior(instance_mode="session") #single instance per proxy connection
class NeuralNetworkServer(object):
    def __init__(self):
        self.initialized = False
        self.record_ids = None
        self.sim = None
        self.pynnx = None

    def set_net(self, simulator_name, description, timestep=1.0, min_delay=1.0,
            per_sim_params={}):
        """Wrapper to execute a PyNN script given a simulator and the network description
        simulator_name: The name of the backend, it will be used to import the correct
            libraries.
        description: A dictionary containing population and projection descriptors.
            Population descriptors must include standard PyNN requirements such as 
            population size, neuron type, parameters (including spike times, rates).
            Projection descriptors must include standard PyNN requirements such as source and target populations, connector type and its parameters, plasticity.
        timestep: Simulation timestep (ms)
        min_delay: Minimum delay in synaptic connections between populations (ms)
        per_sim_params: Extra parameters needed for specific simulator (e.g. 
            wafer id for BrainScaleS, max_neurons_per_core in SpiNNaker)
        """
        self.sim = self.select_simulator(simulator_name) 
        self.pynnx = PyNNAL(self.sim)
        self.pynnx.setup(timestep, min_delay, per_sim_params)
        self.record_ids = None
        # self.description = description
        self.build_populations(description['populations'])
        self.build_projections(description['projections'])

    def select_simulator(self, sim_name):
        """Select proper PyNN backend --- util in __init__.py"""
        return backend_setup(sim_name)

    def build_populations(self, pop_desc):
        """Generate all populations using the PyNN Abstraction Layer (pynnx) to
        avoid ugly code (at least here :P )"""
        pops = {}
        for label in pop_desc:
            _size = pop_desc[label]['size']
            _type = pop_desc[label]['type']
            _params = pop_desc[label]['params']

            pops[label] = self.pynnx.Pop(_size, _type, _params, label)
            if 'record' in pop_desc[label]:
                if self.record_ids is None:
                    self.record_ids = {}
                self.record_ids[label] = pop_desc[label]['record']
                for rec in self.record_ids[label]:
                    self.pynnx.set_recording(pops[label], rec)

        self.populations = pops

    def build_projections(self, proj_desc):
        """Generate all projections using the PyNN Abstraction Layer (pynnx) to
        avoid ugly code (at least here :P )"""
        projs = {}
        for src in proj_desc:
            projs[src] = {}
            for dst in proj_desc[src]:
                _source = self.populations[src]
                _dest = self.populations[dst]
                _conn = proj_desc[src][dst]['conn']
                _w = proj_desc[src][dst]['weights']
                _d = proj_desc[src][dst]['delays'] 
                _tgt = proj_desc[src][dst].get('target', 'excitatory')
                _stdp = proj_desc[src][dst].get('stdp', None)
                _lbl = proj_desc[src][dst].get('label', "{} to {}".format(src, dst))
                _conn_p = proj_desc[src][dst].get('conn_params', {})

                projs[src][dst] = self.pynnx.Proj(_source, _dest, _conn, _w, _d,
                                    _tgt, _stdp, _lbl, _conn_p)
    
    def run(self, run_time, recordings=None):
        if recordings is not None:
            if self.record_ids is None:
                self.record_ids = {}
            for pop_label in recordings:
                self.record_ids[pop_label] = []
                for rec in recordings[pop_label]:
                    self.record_ids[pop_label].append(rec)
                    self.pynnx.set_recording(self.populations[pop_label], rec)

        self.pynnx.run(run_time)
        
    def get_records(self):
        if self.record_ids is None:
            raise Exception("No recordings were set before simulation")

        recs = {}
        for pop_label in self.record_ids:
            recs[pop_label] = {}
            for rec in self.record_ids[pop_label]:
                recs[pop_label][rec] = self.pynnx.get_record( 
                                            self.populations[pop_label], rec)

        return recs
    
    def get_weights(self, weights_to_get):
        _ws = {}
        return _ws

    def end(self):
        self.pynnx.end()

def main():
    Pyro4.Daemon.serveSimple(
            {
                NeuralNetworkServer: "spikevo.pynn_server"
            },
            ns = True, 
            verbose=True, 
            # host="mypynnserver"
        )

if __name__=="__main__":
    main()
