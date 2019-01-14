#!/bin/bash

source ../venv_new/bin/activate
# nAL nKC nLH nDN scale weight_to_spike
python mbody.py 100 2500 20 100 0.025 0.25 \
    --probNoiseSamples=0.1 \
    --probAL=0.2 \
    --probAL2KC=0.15 \
    --nPatternsAL=10 \
    --nSamplesAL=100 \
    --randomizeSamplesAL=1 \
    --renderSpikes=0 \
    --probAL2LH=0.2 \
    --probKC2DN=0.2

