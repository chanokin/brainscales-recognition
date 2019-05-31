from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np

import pynn_genn as sim
from pprint import pprint
from config import *
from utils import *
from multiprocessing import Process, Queue

class Decoder(object):
    def __init__(self, name, params):
        self._network = None
        self.name = name
        self.decode(params)
        print("In Decoder init, %s"%name)

    def decode(self, params):
        net = {}
        net['timestep'] = params.get('timestep', TIMESTEP)
        net['min_delay'] = params.get('min_delay', TIMESTEP)
        net['run_time'] = 1.0#params['run_time']

        pops = {}
        projs = {}

        net['populations'] = pops
        net['projections'] = projs

        pprint(params)

        self._network = net

    def run_pynn(self):
        net = self._network
        pprint(net)
        sim.setup(net['timestep'], net['min_delay'],
                  model_name=self.name,
                  backend='SingleThreadedCPU'
                  )
        for pop in net['populations']:
            pass

        for proj in net['projections']:
            pass

        sim.run(net['run_time'])

        records = {}
        for pop in net['populations']:
            pass

        weights = {}
        for proj in net['projections']:
            pass

        sim.end()


        data = {'recs': records, 'weights': weights}
        return data