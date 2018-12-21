#!/usr/bin/env python
# -*- coding: utf-8; -*-

import sys
import numpy as np

sys.path.append('..') # get prev dir to import input_utils
from input_utils import get_angled_bar as gab
from pprint import pprint
#TODO: make this into real test scripts

fv = 1
thr = 1.0/np.sqrt(2)
thr = None
ih = np.array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [fv, fv, fv, fv, fv],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]) 
pprint(np.array_equal(ih, np.round(gab(5, 1, 0, fv, thr))))

ih = np.array([[0, 0, fv, 0, 0],
               [0, 0, fv, 0, 0],
               [0, 0, fv, 0, 0],
               [0, 0, fv, 0, 0],
               [0, 0, fv, 0, 0]])
pprint(np.array_equal(ih, np.round(gab(5, 1, 90, fv, thr))))

ih = np.array([[0, 0, 0, 0, fv],
               [0, 0, 0, fv, 0],
               [0, 0, fv, 0, 0],
               [0, fv, 0, 0, 0],
               [fv, 0, 0, 0, 0]])
pprint(np.array_equal(ih, np.round(gab(5, 1, 45, fv, thr))))

ih = np.array([[fv, 0, 0, 0, 0],
               [0, fv, 0, 0, 0],
               [0, 0, fv, 0, 0],
               [0, 0, 0, fv, 0],
               [0, 0, 0, 0, fv]])
pprint(np.array_equal(ih, np.round(gab(5, 1, 135, fv, thr))))

ih = np.array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [fv, fv, fv, fv, fv],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]) 
pprint(np.array_equal(ih, np.round(gab(5, 1, 180, fv, thr))))
