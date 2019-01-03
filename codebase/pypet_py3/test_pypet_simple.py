from pypet import Environment
import Pyro4
import Pyro4.util


def run_sim(trajectory):
    print('\n\n')
    print('trajectory.par.synapse.weight = {}'.format(
        trajectory.par.synapse.weight))
    print('\n\n')
    
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

    pynn_server.set_net('nest', description)
    pynn_server.run(trajectory.simulation.duration)
    recs = pynn_server.get_records()
    pynn_server.end()
    
    spike_count = len(recs['destination']['spikes'][0])
    
    trajectory.f_add_result('activity.$', n_spikes=spike_count)

    return len(spike_times)




def post_proc(traj, result_list):
    w_range = traj.par.synapse.f_get('weight')
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
    env = Environment(trajectory='WeightToSpike',
                      comment='Experiment to see which is the minimum weight'
                            'is required by a neuron to spike',
                      add_time=False, # We don't want to add the current time to the name,
                      log_config='DEFAULT',
                      multiproc=True,
                      ncores=2, # Author's laptop had 2 cores XP
                      filename='./hdf5/', # We only pass a folder here, so the name is chosen
                      overwrite_file=True,
                      # automatically to be the same as the Trajectory
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
        'synapse.weight': [0.001, 0.1],
    }
    
    traj.f_explore(explore_dict)

    env.add_postprocessing(post_proc)
    
    env.run(run_sim)

    env.disable_logging()
    
    

if __name__  == '__main__':
    main()


