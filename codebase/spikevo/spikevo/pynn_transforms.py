import numpy as np
import numbers

def is_v9(sim):
    return ('genn' in sim.__name__)

def _ver(sim):
    return (9 if is_v9(sim) else 7)


def Pop(sim, size, cell_class, params, label=None):
    if _ver(sim) == 7:
        return sim.Population(size, cell_class, params, label=label)
    else:
        return sim.Population(size, cell_class(**params), label=label)


def Proj(sim, source_pop, dest_pop, conn_class, weights, delays, 
         target='excitatory', stdp=None, label=None, conn_params={}):
    if _ver(sim) == 7:
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

        return sim.Projection(source_pop, dest_pop, conn,
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


def get_spikes(sim, pop):
    if _ver(sim) == 7:
        data = np.array(pop.getSpikes())
        ids = np.unique(data[:, 0])
        return [data[np.where(data[:, 0] == nid)][:, 1] if nid in ids else []
                for nid in range(pop.size)]
    else:
        data = pop.get_data().segments[0]
        return np.array(data.spiketrains)

def get_weights(sim, proj, format='array'):
    return np.array(proj.getWeights(format=format))
