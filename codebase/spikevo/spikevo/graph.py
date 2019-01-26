from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict
import sys
import numpy as np

import copy

class Node(object):
    def __init__(self, id, is_source=False):
        self.id = id
        self.is_source = is_source
        self.place = np.zeros(2, dtype='uint8')
        self.place_id = -1
        self.outputs = {}

    def dist2(self, node):
        return np.dot(self.place, node.place)

    def connect_to(self, node):
        self.outputs[node.id] = node


class Graph(object):
    def __init__(self):
        self.nodes = {}
        self.sources = {}
        self.inverse_sources = {}
        self.pops = {}

    def add(self, pop, is_source):
        id = pop.label

        if id in self.nodes:
            sys.stderr.write("Population {} has a duplicate label, randomizing!\n".format(id))
            sys.stderr.flush()
            id = "{}_{}".format(id, np.random.randint(0, dtype='uint32'))
            sys.stderr.write("\tnew name {}\n\n".format(id))
            sys.stderr.flush()
            pop.label = id

        self.pops[id] = pop

        if is_source:
            self.sources[id] = Node(id, is_source=is_source)
        else:
            self.nodes[id] = Node(id, is_source=is_source)

        return id

    def plug(self, source_pop, sink_pop):
        if sink_pop.label not in self.nodes:
            raise Exception("Sink population {} has not been registered".format(sink_pop.label))

        elif source_pop.label in self.nodes:
            self.nodes[source_pop.label].connect_to(self.nodes[sink_pop.label])

        elif source_pop.label in self.sources:
            self.inverse_sources[sink_pop.label] = source_pop.label
            self.sources[source_pop.label].connect_to(self.nodes[sink_pop.label])

        else:
            raise Exception("Source population {} has not been registered\n\nNodes: {}\n\nSources: {}".\
                        format(source_pop.label, self.nodes.keys(), self.sources.keys()))



    def clone(self):
        new_graph = Graph()
        new_graph.nodes = copy.deepcopy(self.nodes)
        new_graph.sources = copy.deepcopy(self.sources)
        new_graph.pops = self.pops

        return new_graph

    def evaluate(self, distances, mapping):
        dist2 = 0.0
        for src in self.nodes:
            targets = self.nodes[src].outputs
            for tgt in targets:
                idx_src = mapping[self.nodes[src].place_id]
                idx_tgt = mapping[self.nodes[tgt].place_id]
                dist2 += distances[idx_src, idx_tgt]

        return dist2

    def update_places(self, places):
        for _id in sorted(places):
            self.nodes[_id].place[:] = places[_id]
