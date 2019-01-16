from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict
import sys
import numpy as np

class Node(object):
    def __init__(self, id, is_source=False):
        self.id = id
        self.is_source = is_source
        self.place = np.zeros((1, 1))
        self.outputs = {}

    def dist2(self, node):
        return np.dot(self.place, node.place)

    def connect_to(self, node):
        self.outputs[node.id] = node


class Graph(object):
    def __init__(self):
        self.nodes = {}
        self.sources = {}
        self.pops = {}

    def add(self, pop, is_source):
        id = pop.label

        if id in self.nodes:
            sys.stderr.write("Population {} has a duplicate label, randomizing!\n".format(id))
            sys.stderr.flush()
            id = "{}_{}".format(id, np.random.randint(0, dtype='uint32'))
            sys.stderr.write("\tnew name {}\n\n".format(id))
            sys.stderr.flush()

        self.nodes[id] = Node(id, is_source=is_source)
        self.pops[id] = pop

        if is_source:
            self.sources[Node]

        return id

    def plug(self, source_pop, sink_pop):
        if source_pop.label not in self.nodes:
            raise Exception("Source population {} has not been registered".format(source_pop.label))
        if sink_pop.label not in self.nodes:
            raise Exception("Sink population {} has not been registered".format(sink_pop.label))

        self.nodes[source_pop.label].connect_to(self.nodes[sink_pop.label])
