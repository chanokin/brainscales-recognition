from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np
from pprint import pprint
from multiprocessing import Process, Queue
from snn_decoder import Decoder
import time

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
            t0 = time.time()
            dec = Decoder(label, description)
            data = dec.run_pynn()
            t1 = time.time()
            t = t1-t0
            minutes = np.floor(t/60.)
            seconds = np.round(t - minutes*60.0, decimals=2)
            print("\n\n------------------------------------------------")
            print("\n\nexperiment took {} minutes {} seconds".format(minutes, seconds))
            print("\n\n------------------------------------------------\n")
            queue.put(data)

        q = Queue()
        p = Process(target=f, args=(q, network_description))
        p.start()
        data = q.get()
        # pprint(label)
        # pprint(data)
        self._processes[label] = p

        return data

