import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pyNN.hardware.brainscales as pynn
#import pyNN.hardware.facets as pynn

USE_EMU = bool(1)
pynn.setup(useSystemSim=USE_EMU)

neurons = pynn.Population(1, pynn.IF_cond_exp, 
            {'cm': 0.2,
             'e_rev_I': -100.0,
             'v_rest': -50.0,}
          )
neurons.record()
inputs = pynn.Population(1, pynn.SpikeSourcePoisson, {'rate': 10.0})

proj = pynn.Projection(inputs, neurons,
        pynn.OneToOneConnector(weights=15.0))

pynn.run(1000)

out_spikes = np.array(neurons.getSpikes())

pynn.end()

plt.figure()
plt.plot(out_spikes[:, 1], out_spikes[:, 0], '.')
plt.savefig("output.pdf")




