from __future__ import (print_function,
                        unicode_literals,
                        division)
                        
from pypet import Environment
import Pyro4
import Pyro4.util

SIM_NAME = 'genn'

def run_sim(trajectory):
    e_rev = 50
    neuron_parameters = {
    'cm': 0.2,
    'v_reset': -70.,
    'v_rest': -50.,
    'v_thresh': -25.,
    'e_rev_I': -e_rev,
    'e_rev_E': e_rev,
    'tau_m': 10.,
    'tau_refrac': 0.1,
    'tau_syn_E': 5.,
    'tau_syn_I': 5.,
    }

    description = {
        'populations': {
            'source': {
                'type': 'SpikeSourceArray',
                'size': 1,
                'params': { 
                    'spike_times': [10],
                },
                # 'record': ['spikes'],
            },
            'destination': {
                'type': 'IF_cond_exp',
                'size': 1,
                'params': neuron_parameters,
                'record': ['spikes']
            }
        },
        'projections':{
            'source': {
                'destination': {
                    'conn': 'OneToOneConnector',
                    'weights': trajectory.par.synapse.weight,
                    'delays': 1.0,
                }
            }
        }
    }
    
    pynn_server = Pyro4.Proxy("PYRONAME:spikevo.pynn_server")

    strw = '{}'.format(trajectory.par.synapse.weight)
    strw = strw.replace('.', 'p')

    pynn_server.set_net(SIM_NAME, description, 
        per_sim_params={'model_name': 'pyro_pynn_{}'.format(strw)})
        
    print('\n\n---------------------------------------\n')
    print('after set_net ({})'.format(strw))
    print('\n---------------------------------------\n')

    pynn_server.run(trajectory.simulation.duration)
    print('\n\n---------------------------------------\n')
    print('after run ({})'.format(strw))
    print('\n---------------------------------------\n')
    
    recs = pynn_server.get_records()
    print('\n\n---------------------------------------\n')
    print('after get_records ({})'.format(strw))
    print('\n---------------------------------------\n')
    
    pynn_server.end()
    print('\n\n---------------------------------------\n')
    print('after end ({})'.format(strw))
    print('\n---------------------------------------\n')
    
    spike_count = len(recs['destination']['spikes'][0])

    trajectory.f_add_result('activity.$', n_spikes=spike_count)

    print('\n\n---------------------------------------\n')
    print('after add result ({})'.format(strw))
    print('\n---------------------------------------\n')
    return spike_count




def post_proc(traj, result_list):
    w_range = traj.par.synapse.f_get('synapse.weight')
    for result_tuple in result_list:
        run_idx = result_tuple[0]
        n_spikes = result_tuple[1]
        
        w_val = w_range[run_idx]
        
        print(run_idx, w_val, n_spikes)
        

######################################################################
######################################################################
######################################################################

def main():
    ### setup an experimental environment
    multiproc = False if SIM_NAME == 'genn' else True
        
    env = Environment(trajectory='WeightToSpike',
                      comment='Experiment to see which is the minimum weight'
                            'is required by a neuron to spike',
                      add_time=False, # We don't want to add the current time to the name,
                      log_config='DEFAULT',
                      multiproc=multiproc,
                      ncores=2, # Author's laptop had 2 cores XP
                      filename='./hdf5/', # We only pass a folder here, so the name is chosen
                      overwrite_file=True,
                      ### from the Brian2 example
                      continuable=False,
                      lazy_debug=False,
                      # use_pool=False, # We cannot use a pool, our network cannot be pickled
                      # wrap_mode='QUEUE',
                      )


    ### Get the trajectory object for the recently created envirnoment
    traj = env.trajectory

    ### Set simulation time
    traj.f_add_parameter('simulation.duration', 100.0, #ms
        comment='how long time to simulate the neural network')
    traj.f_add_parameter('synapse.weight', 0.001,
        comment='weight to explore')
    ### exploration dictionary
    ### describe parameters with certain rules (e.g. cartesian product of params)
    explore_dict = {
        'synapse.weight': [0.001, 0.1], #values to explore
    }
    
    traj.f_explore(explore_dict)

    env.add_postprocessing(post_proc)
    
    env.run(run_sim)

    env.disable_logging()
    
    

if __name__  == '__main__':
    main()


