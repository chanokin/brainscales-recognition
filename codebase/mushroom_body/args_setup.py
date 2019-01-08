from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Mushroom body experiment for classification')

    parser.add_argument('nAL', type=int,
        help='Number of neurons in the Antennae Lobe (first layer)' )
    parser.add_argument('nKC', type=int,
        help='Number of (Kenyon) neurons in the Mushroom body (second layer)' )
    parser.add_argument('nLH', type=int,
        help='Number of inhibitory interneurons in the Lateral Horn (second layer)' )
    parser.add_argument('nDN', type=int,
        help='Number of decision neurons (ouput [third] layer)' )
    parser.add_argument('gScale', type=float,
        help='Global rescaling factor for synaptic strength (weights)' )
    
    parser.add_argument('--probAL', type=float, default=0.2,
        help='Probability of active cells in the Antennae Lobe')
    parser.add_argument('--nPatternsAL', type=int, default=10,
        help='Number of patterns for the Antennae Lobe')
    parser.add_argument('--nSamplesAL', type=int, default=100,
        help='Number of samples from the patterns of the Antennae Lobe')
    parser.add_argument('--probNoiseSamplesAL', type=float, default=0.1,
        help='Probability of neurons in the Antennae Lobe flipping state')
    parser.add_argument('--probAL2KC', type=float, default=0.1,
        help='Probability of connectivity between the Antennae Lobe and the ')
    
    

    return parser.parse_args()
