from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict

import numpy as np

from .wafer_blacklists import BLACKLISTS
from .wafer import Wafer as WAL


class WaferPlacer(object):
    def __init__(self, graph, wafer, constraints={}):
        self._graph = graph
        self._wafer = wafer
        self._constraints = constraints

        self._set_constraints()
        # self._place()



    def _set_constraints(self):
        pass

    def _place(self):
        pass
