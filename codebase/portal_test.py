import pyhmf as pynn
from pymarocco import PyMarocco

marocco = PyMarocco()
marocco.backend = PyMarocco.Hardware

pynn.setup(marocco=marocco)

neuron = pynn.Population(1, pynn.IF_cond_exp, {})
inputs = pynn.Population(1, sim.SpikeSourcePoisson(rate=10.0))

pynn.run(10)

pynn.end()

