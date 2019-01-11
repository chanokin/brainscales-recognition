#!/bin/bash

source ../venv_new/bin/activate
# nAL nKC nLH nDN scale weight_to_spike
python mbody.py 100 1000 20 100 0.025 1.0 \
    --probNoiseSamples=0.1 \
    --probAL=0.2 \
    --probAL2KC=0.15 \
    --nPatternsAL=10 \
    --nSamplesAL=2000 \
    --randomizeSamplesAL=1 \
    --renderSpikes=0 \
    --probAL2LH=0.2

