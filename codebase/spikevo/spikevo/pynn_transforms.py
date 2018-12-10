def _ver(sim):
    return int(sim.__version__.split('.')[0])


def Pop(sim, size, cell_class, params, label=None):
    if _ver(sim) == 7:
        return Population(size, cell_class, params, label=label)
    else:
        return Population(size, cell_class(**params), label=label)


def Proj(sim, source_pop, dest_pop, conn_class, weights, delays, 
         target, stdp=None, label=None, conn_params=None):
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
        if stdp is not None
            synapse = sim.STDPMechanism(
                timing_dependence=stdp['timing_dependence'],
                weight_dependence=stdp['weight_dependence'],
                weight=weights, delay=delays)
        else:
            synapse = sim.StaticSynapse(weight=weights, delay=delays)

        return sim.Projection(source_pop, dest_pop, conn_class(**conn_params), 
                synapse, receptor_type=target, label=label)


