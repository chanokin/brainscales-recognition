from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

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

            return self._graph.evaluate()


        clean_ids, clean_coords = self._wafer.available()
        # print(clean_ids)
        # print(clean_coords)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        IND_SIZE = len(self._graph.nodes)

        toolbox = base.Toolbox()
        # np.random.choice args: (set from which to choose, sample size, with replacement? )
        toolbox.register("attr_indices", np.random.choice, clean_ids, IND_SIZE, False)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        p = toolbox.population(n=2)
        print(p)
    @property
    def places(self):
        return {i: self._graph.nodes[i].place for i in sorted(self._graph.nodes)}

