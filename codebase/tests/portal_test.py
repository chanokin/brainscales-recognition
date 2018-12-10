#!/usr/bin/env python
# coding: utf-8
"""
BrainScaleS system example

This script creates a chain of neurons which ideally should 'transport' the
spike from one neuron to the next.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wafer", type=int)
parser.add_argument("--hicann", type=int)
import numpy as np
import gzip
#import cPickle as serializer
import json as serializer
from quantities import Hz, s, ms, mV

import pyhmf as pynn
from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pymarocco import PyMarocco, Defects

args = parser.parse_args()
wafer = args.wafer
hicann = args.hicann

def init_logging():
    from pysthal.command_line_util import init_logger
    init_logger("WARN", [
        ("guidebook", "ERROR"),
        ("marocco", "ERROR"),
        ("Calibtic", "ERROR"),
        ("sthal", "ERROR")
    ])

    import pylogging
    logger = pylogging.get("guidebook")
    return logger

def setup_marocco(wafer):
    from pymarocco.runtime import Runtime
    from pymarocco.coordinates import LogicalNeuron
    from pymarocco.results import Marocco

    marocco = PyMarocco()
    marocco.neuron_placement.default_neuron_size(4)
    marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
    marocco.merger_routing.strategy(marocco.merger_routing.one_to_one)

    marocco.bkg_gen_isi = 125
    marocco.pll_freq = 125e6

    marocco.backend = PyMarocco.Hardware
    marocco.calib_backend = PyMarocco.Binary
    marocco.defects.path = marocco.calib_path = "/wang/data/calibration/brainscales/default"
    marocco.defects.backend = Defects.XML
    marocco.default_wafer = C.Wafer(wafer)
    marocco.param_trafo.use_big_capacitors = True
    marocco.input_placement.consider_firing_rate(True)
    marocco.input_placement.bandwidth_utilization(0.8)
    marocco.verification = PyMarocco.Skip
    marocco.checkl1locking = PyMarocco.SkipCheck
    
    runtime = Runtime(marocco.default_wafer)
    return marocco, runtime

def map_to_hardware(marocco, runtime, duration):
    marocco.skip_mapping = False
    marocco.backend = PyMarocco.None

    pynn.reset()
    pynn.run(duration)

    def set_sthal_params(wafer, gmax, gmax_div):
        for hicann in wafer.getAllocatedHicannCoordinates():
            fgs = wafer[hicann].floating_gates
            for ii in xrange(fgs.getNoProgrammingPasses()):
                cfg = fgs.getFGConfig(C.Enum(ii))
                cfg.fg_biasn = 0
                cfg.fg_bias = 0
                fgs.setFGConfig(C.Enum(ii), cfg)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
                fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)

            for driver in C.iter_all(C.SynapseDriverOnHICANN):
                for row in C.iter_all(C.RowOnSynapseDriver):
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.left, gmax_div)
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.right, gmax_div)

    set_sthal_params(runtime.wafer(), gmax=1023, gmax_div=1)

    marocco.skip_mapping = True
    marocco.backend = PyMarocco.Hardware
    marocco.hicann_configurator = PyMarocco.ParallelHICANNv4Configurator

def get_record_runs(pop, neurons_in_pop, runtime):
    record_runs = []
    record_idx_all = []
    for nrn_idx in range(neurons_in_pop):

        record_nrn_idxs = sorted([nrn_idx, (nrn_idx + 1) % neurons_in_pop])
        record_idx_all.append(record_nrn_idxs)

        record = []
        for n, neuron in enumerate(pop):
            if n not in record_nrn_idxs:
                continue
            item = runtime.results().placement.find(neuron)[0]
            logical_neuron = item.logical_neuron()
            record.append(logical_neuron)
        record_runs.append(record)
    return record_runs, record_idx_all

def record_logical_neuron(logical_neuron, runtime):
    runtime.results().analog_outputs.record(logical_neuron)

def save_to_file(data, filepath):
    """
    Args:
        data (dict)
        filepath (str)
    """
    #with open(filepath, 'wb') as handle:
    with gzip.GzipFile(filepath + ".gz", 'w') as handle:
        serializer.dump(data, handle)

def record_neuron(pop, index_list, runtime):
    """
    record the trace of neurons in 'pop' with indices 'index_list'
    """
    logical_neurons = []
    for n, neuron in enumerate(pop):
        if n not in index_list:
            continue
        item = runtime.results().placement.find(neuron)[0]
        logical_neuron = item.logical_neuron()
        runtime.results().analog_outputs.record(logical_neuron)
        logical_neurons.append(logical_neuron)
    return logical_neurons

def unrecord_logical_neurons(logical_neurons, runtime):
    for logical_neuron in logical_neurons:
        runtime.results().analog_outputs.unrecord(logical_neuron)

def get_spikes(pop, neurons_in_pop):
    spikes_all = []
    for spike_nrn_idx in range(neurons_in_pop):
        spikes = pynn.PopulationView(pop, [spike_nrn_idx]).getSpikes()
        spike_times = list(spikes[:,1])
        spikes_all.append(spike_times)
    return spikes_all

def get_traces(pop):
    voltage_array = pop.get_v()
    traces = []
    for i in sorted(set(voltage_array[:,0])):
        voltage_i = voltage_array[voltage_array[:,0] == i]
        trace = [[t,v] for _, t, v in voltage_i]
        traces.append(trace)
    return traces


logger = init_logging()
# marocco is the object responsible for the mapping, runtime holds all mapping data
marocco, runtime = setup_marocco(wafer)
pynn.setup(marocco=marocco, marocco_runtime=runtime)

neuron_parameters = {
    'cm': 0.2,
    'v_reset': -30.,
    'v_rest': -20.,
    'v_thresh': -15,
    'e_rev_I': -100.,
    'e_rev_E': 60.,
    'tau_m': 5.,
    'tau_refrac': 0.5,
    'tau_syn_E': 5.,
    'tau_syn_I': 5.,
}

## adjustable variables
# duration of the emulation
duration = 1500.0
# number of neurons in the chain
neurons_in_pop = 30
# spike times in ms that excite the first neuron
exc_spike_times = [50, 51]
# spike times in ms that inhibit ALL neurons
inh_spike_times = [0]
# connect neuron n+1 with neuron n inhibitorily
backward_inhibition = False
# connect neuron n_max to neuron 0
cyclic = False

pop = pynn.Population(neurons_in_pop, pynn.IF_cond_exp, neuron_parameters)
pop.record() # record spikes of all neurons in pop

# place the neurons we defined in software on the BrainScaleS hardware
marocco.manual_placement.on_hicann(pop, C.HICANNOnWafer(C.Enum(hicann)))

stimulus_exc = pynn.Population(1, pynn.SpikeSourceArray, {
    'spike_times': exc_spike_times})
stimulus_inh = pynn.Population(1, pynn.SpikeSourceArray, {
    'spike_times': inh_spike_times})

connector = pynn.AllToAllConnector(weights=1)
projections = [
    pynn.Projection(stimulus_exc, pynn.PopulationView(pop,[0]), connector, target='excitatory'),
    pynn.Projection(stimulus_inh, pop, connector, target='inhibitory')
]


for nrn in range(neurons_in_pop-1):
    print "from {} to {}".format(nrn, nrn+1)
    for _ in range(2):
        pynn.Projection(pynn.PopulationView(pop, [nrn]),
                        pynn.PopulationView(pop, [nrn+1]),
                        connector,
                        target="excitatory")
        if nrn > 0 and backward_inhibition:
            pynn.Projection(pynn.PopulationView(pop, [nrn]),
                            pynn.PopulationView(pop, [nrn-1]),
                            connector,
                            target="inhibitory")

if cyclic:
    for _ in range(4):
        pynn.Projection(pynn.PopulationView(pop, [neurons_in_pop-1]),
                        pynn.PopulationView(pop, [0]),
                        connector,
                        target="excitatory")

if backward_inhibition:
    for _ in range(1):
        pynn.Projection(pynn.PopulationView(pop, [0]),
                        pynn.PopulationView(pop, [neurons_in_pop-1]),
                        connector,
                        target="inhibitory")

map_to_hardware(marocco, runtime, duration)

# record_runs contains a list of logical_neuron objects which represent the hardware neurons
record_runs, record_idxs = get_record_runs(pop, neurons_in_pop, runtime)

# here the experiment is executed
data_all = {}
for idx, logical_neurons in enumerate(record_runs):

    for logical_neuron in logical_neurons:
        record_logical_neuron(logical_neuron, runtime)

    pynn.run(duration)

    traces = get_traces(pop)
    spikes = get_spikes(pop, neurons_in_pop)
    data_all[idx] = dict(traces=traces, spikes=spikes, idxs=record_idxs[idx])

    pynn.reset()
    unrecord_logical_neurons(logical_neurons, runtime)
    # only change neuron parameters from now on
    marocco.hicann_configurator = PyMarocco.OnlyNeuronNoResetNoFGConfigurator

save_to_file(data_all, 'chain_data.pkl')
