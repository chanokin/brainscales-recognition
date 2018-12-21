#!/usr/bin/env python
# -*- coding: utf-8; -*-

import sys
import numpy as np

sys.path.append('..') # get prev dir to import input_utils
from input_utils import get_receptive_fields_indices as grsi
from pprint import pprint

def test_dict(hand, comp):
    result = True
    for r in hand:
        for c in hand[r]:
            result &= np.array_equal(hand[r][c], comp[r][c])
            if result == False:
                return result

    return result

dh = {1: {1: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])}}
dc = grsi(3, 3, 3, 1)
pprint(dh)
pprint(dc)
pprint(test_dict(dh, dc))


dh = {
    1: {
        1: np.array([0, 1, 2, 4, 5, 6, 8, 9, 10]),
        2: np.array([1, 2, 3, 5, 6, 7, 9, 10, 11])
    }
}
dc = grsi(4, 3, 3, 1)
pprint(dh)
pprint(dc)
pprint(test_dict(dh, dc))


dh = {
    2: {
        2: np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 
                     14, 15, 16, 17, 18, 21, 22, 23, 24, 25,
                     28, 29, 30, 31, 32]),
        4: np.array([2, 3, 4, 5, 6, 9, 10, 11, 12, 13,
                     16, 17, 18, 19, 20, 23, 24, 25, 26, 27,
                     30, 31, 32, 33, 34])
    }
}
dc = grsi(7, 5, 5, 2)
pprint(dh)
pprint(dc)
pprint(test_dict(dh, dc))
