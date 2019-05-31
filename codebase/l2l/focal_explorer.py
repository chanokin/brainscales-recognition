from pypet import Environment, cartesian_product

import random
from deap import base
from deap import creator
from deap import tools
from pprint import pprint
import numpy as np
from config import *
from snn_executor import Executor


def eval_one_min(trajectory):
    
    populations = {
    }

    # 'conn':, 'weights':, 'delays':, 'conn_params',
    projections = {
    }

    description = {
        'populations': populations,
        'projections': projections,
    }
    
    individual = trajectory.parameters.ind_idx
    generation = trajectory.parameters.generation
    name = 'gen{}_ind{}'.format(generation, individual)
    net_params = {attr: trajectory.individual[i] for attr, i in ATTR2IDX.items()}
    # print("\n\n%s"%(name))
    # pprint(net_params)

    ex = Executor()
    data = ex.run(name, net_params)

    trajectory.f_add_result('activity.$', data=data)

    return (sum(trajectory.individual),)




######################################################################
######################################################################
######################################################################

def main():
    ### setup an experimental environment
    multiproc = True
    n_procs = 1
    env = Environment(trajectory='WeightToSpike',
                      comment='Experiment to see which is the minimum weight'
                            'is required by a neuron to spike',
                      add_time=False, # We don't want to add the current time to the name,
                      # log_config='DEFAULT',
                      log_level=50,  # only display ERRORS
                      multiproc=multiproc,
                      ncores=n_procs, # Author's laptop had 2 cores XP
                      filename='./hdf5/', # We only pass a folder here, so the name is chosen
                      overwrite_file=True,
                      ### from the Brian2 example
                      continuable=False,
                      lazy_debug=False,
                      # use_pool=False, # We cannot use a pool, our network cannot be pickled
                      # wrap_mode='QUEUE',
                      
                      ### from DEAP example
                      automatic_storing=False,  # This is important, we want to run several
                      # batches with the Environment so we want to avoid re-storing all
                      # data over and over again to save some overhead.
                      )

    ### Get the trajectory object for the recently created envirnoment
    traj = env.trajectory

    ### genetic algorithm parameters
    traj.f_add_parameter('popsize', 3, comment='Population size')
    traj.f_add_parameter('CXPB', 0.5, comment='Crossover term')
    traj.f_add_parameter('MUTPB', 0.2, comment='Mutation probability')
    traj.f_add_parameter('NGEN', 2, comment='Number of generations')

    traj.f_add_parameter('generation', 0, comment='Current generation')
    traj.f_add_parameter('ind_idx', 0, comment='Index of individual')
    
    ### in our case we only optimize a single weight, so length == 1
    ### we need at least 2?
    traj.f_add_parameter('ind_len', 1, comment='Length of individual')

    traj.f_add_parameter('indpb', 0.05, comment='Mutation parameter')
    traj.f_add_parameter('tournsize', 3, comment='Selection parameter')

    traj.f_add_parameter('seed', 42, comment='Seed for RNG')
    
    traj.f_add_parameter('simulation.duration', 100.0)#ms
    
    # Placeholders for individuals and results that are about to be explored
    traj.f_add_derived_parameter('individual', [0 for x in range(traj.ind_len)],
                                 'An indivudal of the population')
    traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')
    
    ### setup DEAP minimization 
    ### Name of our fitness function, base class from which to inherit,
    ### weights are the importance of each element in the fitness function
    ### return values; -1.0 because it's a minimization problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    
    ### Name of our individual object, it inherits from a list and has a
    ### fitness attribute which points to the recently created fitness func
    creator.create("Individual", list, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    # Attribute generator

    attr_list = [None for _ in ATTR2IDX]
    for attr in ATTR2IDX:
        r = ATTR_RANGES[attr]
        f = np.random.uniform if np.issubdtype(type(r[0]), np.floating) else randint_float
        toolbox.register(attr, f, r[0], r[1])
        attr_list[ATTR2IDX[attr]] = getattr(toolbox, attr)

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     attr_list, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=traj.indpb)
    toolbox.register("select", tools.selTournament, tournsize=traj.tournsize)
    toolbox.register("evaluate", eval_one_min)
    toolbox.register("map", env.run)  # We pass the individual as part of traj, so


    # ------- Initialize Population -------- #
    random.seed(traj.seed)

    pop = toolbox.population(n=traj.popsize)
    CXPB, MUTPB, NGEN = traj.CXPB, traj.MUTPB, traj.NGEN
    
    print("Start of evolution")
    for g in range(traj.NGEN):

        # ------- Evaluate current generation -------- #
        print("-- Generation %i --" % g)

        # Determine individuals that need to be evaluated
        eval_pop = [ind for ind in pop if not ind.fitness.valid]

        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        # prod = cartesian_product({'generation': [g],
        #                           'ind_idx': range(len(eval_pop))})
        prod = cartesian_product({'generation': [g],
                                  'ind_idx': range(len(eval_pop)),
                                  'individual':[list(x) for x in eval_pop]},
                                  [('ind_idx', 'individual'),'generation'])
        traj.f_expand(prod)

        fitnesses_results = toolbox.map(toolbox.evaluate)  # evaluate using our fitness function

        # fitnesses_results is a list of
        # a nested tuple: [(run_idx, (fitness,)), ...]
        for idx, result in enumerate(fitnesses_results):
            # Update fitnesses
            _, fitness = result  # The environment returns tuples: [(run_idx, run), ...]
            eval_pop[idx].fitness.values = fitness

        # Append all fitnesses (note that DEAP fitnesses are tuples of length 1
        # but we are only interested in the value)
        pprint([x.fitness.values for x in eval_pop])

        # Gather all the fitnesses in one list and print the stats
        fits = [x.fitness.values[0] if len(x.fitness.values) else 0 for x in eval_pop]
        traj.fitnesses.extend(fits)

        print("  Evaluated %i individuals" % len(fitnesses_results))

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


        # ------- Create the next generation by crossover and mutation -------- #
        if g < traj.NGEN -1:  # not necessary for the last generation
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # The population is entirely replaced by the offspring
            pop[:] = offspring

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    traj.f_store()  # We switched off automatic storing, so we need to store manually
    
    

if __name__  == '__main__':
    main()


