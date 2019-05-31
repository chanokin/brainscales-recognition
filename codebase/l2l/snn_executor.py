from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np

import pynn_genn as sim
from pprint import pprint

from multiprocessing import Process, Queue

class Decoder(object):
    def __init__(self, name, params):
        self.decode(params)
        self.name = name
        print("In Decoder init, %s"%name)

    def decode(self, params):
        pprint(params)

class Executor(object):
    def __init__(self):
        self._processes = {}

    def __del__(self):
        for p in self._processes:
            try:
                self._processes[p].join()
            except:
                print("failed to join process %s"%(p))
    def run(self, label, network_description):
        def f(queue, description):
            x = 123
            dec = Decoder(label, description)
            queue.put([x])

        q = Queue()
        p = Process(target=f, args=(q, network_description))
        p.start()
        data = q.get()
        self._processes[label] = p

        return data