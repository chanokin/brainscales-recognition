import pyhmf as pynn
from pymarocco import PyMarocco

marocco = PyMarocco()
marocco.backend = PyMarocco.Hardware

pynn.setup(marocco=marocco)

neurons = pynn.Population(1, pynn.IF_cond_exp, {})
inputs = pynn.Population(1, sim.SpikeSourcePoisson(rate=10.0))

proj = pynn.Projection(neurons, inputs, 
        pynn.OneToOneConnector(5.0))

pynn.run(10)

pynn.end()

