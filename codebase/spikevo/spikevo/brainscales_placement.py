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

        def evaluate(individual):
            keys = self._graph.nodes.keys()
            for i, hicann in enumerate(individual):
                self._graph.nodes[keys[i]].place[:] = self._wafer.i2c(hicann)
                self._graph.nodes[keys[i]].place_id = hicann

            evaluation = self._graph.evaluate(self._wafer.distances, self._wafer.id2idx)
            unique_error = len(individual) - len(np.unique(individual))
            # print(evaluation)
            return (evaluation, unique_error)

        def cxTwoPointCopy(ind1, ind2):
            """Execute a two points crossover with copy on the input individuals. The
            copy is required because the slicing in numpy returns a view of the data,
            which leads to a self overwritting in the swap operation.
            """
            size = len(ind1)
            cxpoint1 = np.random.randint(1, size)
            cxpoint2 = np.random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            nind1 = ind1.copy()
            nind2 = ind2.copy()
            nind1[cxpoint1:cxpoint2], nind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

            # u1, c1 = np.unique(nind1, return_counts=True)
            # if len(u1) != len(nind1):
            #     for i, c in enumerate(c1):
            #         if c > 1:
            #             whr = np.where(nind1 == c)
            #             for j, w in enumerate(whr):
            #                 nind1[w] += j
            #     print(ind1, nind1, ind2, nind2)

            ind1[:] = nind1
            ind2[:] = nind2
            return ind1, ind2

        width = self._graph.width
        height = self._graph.height

        clean_ids, clean_coords = self._wafer.available(width, height)
        print(clean_ids)
        print(clean_coords)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1000.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        IND_SIZE = len(self._graph.nodes)

        toolbox = base.Toolbox()
        # np.random.choice args: (set from which to choose, sample size, with replacement? )
        toolbox.register("attr_indices", np.random.choice, clean_ids, IND_SIZE, False)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", cxTwoPointCopy)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=1000)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop1, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                                        halloffame=hof)
        # print(pop1)
        # print(log)
        print(hof)
        print(np.sort(hof))

    @property
    def places(self):
        return {i: self._graph.nodes[i].place for i in sorted(self._graph.nodes)}

