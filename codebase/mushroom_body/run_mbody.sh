#!/bin/bash

source ../venv_new/bin/activate
# nAL nKC nLH nDN scale weight_to_spike
python mbody.py 100 2500 20 100 0.025 0.0025 \
    --probNoiseSamples=0.1 \
    --probAL=0.2 \
    --probAL2KC=0.15 \
    --nPatternsAL=10 \
    --nSamplesAL=5000 \
    --randomizeSamplesAL=1 \
    --renderSpikes=0 \
    --probKC2DN=0.2

