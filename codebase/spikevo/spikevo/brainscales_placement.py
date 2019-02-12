from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import os


class WaferPlacer(object):
    def __init__(self, graph, wafer, constraints={}):
        self._graph = graph.clone()
        self._wafer = wafer.clone()
        self._constraints = constraints
        self._set_constraints()
        # self._place()



    def _set_constraints(self):
        pass

    def _place(self):

        def new_individual(indices, size):
            return np.random.choice(indices, size, False)

        def evaluate(individual):
            keys = self._graph.nodes.keys()
            for i, hicann in enumerate(individual):
                self._graph.nodes[keys[i]].place[:] = self._wafer.i2c(hicann)
                self._graph.nodes[keys[i]].place_id = hicann

            eval, sub_eval = self._graph.evaluate(self._wafer.distances, self._wafer.id2idx, subpop_weight=1.0)
            unique_error = len(individual) - len(np.unique(individual))
            # print(len(individual), len(np.unique(individual)), len(individual) - len(np.unique(individual)))
            # print(evaluation, unique_error)
            return eval, sub_eval

        def cxTwoPointCopy(ind1, ind2, indices):
            """Execute a two points crossover with copy on the input individuals. The
            copy is required because the slicing in numpy returns a view of the data,
            which leads to a self overwritting in the swap operation.
            """
            # np.random.seed()
            size = len(ind1)
            cxpoint1 = np.random.randint(1, size)
            cxpoint2 = np.random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            nind1 = ind1.copy()
            nind2 = ind2.copy()
            nind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy()
            nind2[cxpoint1:cxpoint2] = ind1[cxpoint1:cxpoint2].copy()
            # print("pre-uniqued", sorted(nind1), sorted(nind2))

            unique, counts = np.unique(nind1, return_counts=True)
            for i, c in enumerate(counts):
                if c > 1:
                    whr = np.where(nind1 == unique[i])[0]
                    for j, w in enumerate(whr):
                        valid_indices = np.setdiff1d(indices, nind1)
                        nind1[w] = np.random.choice(valid_indices)


            unique, counts = np.unique(nind2, return_counts=True)
            for i, c in enumerate(counts):
                if c > 1:
                    whr = np.where(nind2 == unique[i])[0]
                    for j, w in enumerate(whr):
                        valid_indices = np.setdiff1d(indices, nind2)
                        nind2[w] = np.random.choice(valid_indices)


            # print("post-uniqued", sorted(nind1), sorted(nind2))

            ind1[:] = nind1
            ind2[:] = nind2
            return ind1, ind2

        def mutIndividual(individual, indices, indpb):
            np.random.seed()
            for i in range(len(individual)):
                valid_indices = np.setdiff1d(indices, individual)
                if np.random.uniform(0, 1) < indpb:
                    individual[i] = np.random.choice(valid_indices)

            return individual,

        np.random.seed()

        width = self._graph.width + 4
        height = self._graph.height + 4

        clean_ids, clean_coords = self._wafer.available(width, height)
        print("clean_ids", clean_ids)
        print("clean_coords", clean_coords)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        IND_SIZE = len(self._graph.nodes)

        toolbox = base.Toolbox()
        # np.random.choice args: (set from which to choose, sample size, with replacement? )
        toolbox.register("attr_indices", np.random.choice, clean_ids, IND_SIZE, False)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", cxTwoPointCopy, indices=clean_ids)
        toolbox.register("mutate", mutIndividual, indices=clean_ids, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop1, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=500, stats=stats,
                                        halloffame=hof)
        # pop1, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=300, cxpb=0.5, mutpb=0.5,
        #                  ngen=1000, stats=stats, halloffame=hof)
        # pop1, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.2, mutpb=0.2,
        #                  ngen=100, stats=stats, halloffame=hof)
        # print(pop1)
        # print(log)

        import matplotlib.pyplot as plt
        from pprint import pprint

        keys = self._graph.nodes.keys()
        for ind in pop1:
            print(ind.fitness)

        for ind in hof:
            print(ind.fitness)
            for i, hicann in enumerate(np.sort(ind)):
                # print(i, hicann)
                self._graph.nodes[keys[i]].place[:] = self._wafer.i2c(hicann)
                self._graph.nodes[keys[i]].place_id = hicann

            # evaluation = self._graph.evaluate(self._wafer.distances, self._wafer.id2idx, subpop_weight=2.0)
            # unique_error = float( len(ind) - len(np.unique(ind)) )
            # print((evaluation, unique_error))

        plc = self.places
        pprint(plc)

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        for k in plc:
            color = 'r' if k.startswith('neurons') else 'b'
            plt.plot(plc[k][1], plc[k][0], 's', color=color)

        ax.set_xlim(-1, self._wafer._width + 1)
        ax.set_ylim(self._wafer._height + 1, -1)

        plt.show()
        os.sys.exit()

    @property
    def places(self):
        return {i: self._graph.nodes[i].place for i in sorted(self._graph.nodes)}

