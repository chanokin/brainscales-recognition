#!/bin/bash

IS_NVIDIA=`lspci | grep VGA | grep -i nvidia`

if [ -z "$IS_NVIDIA" ]
then
    echo "nVidia GPU not found!"
    echo 
    exit 0
fi

HAS_CUDA=`ls /usr/local | grep -i cuda`

HAS_CUDA=$HAS_CUDA || `ls /usr/bin | grep nvcc`

if [ -z "$HAS_CUDA" ]
then
    echo "CUDA not found, please install?"
    echo 
    exit 0
fi

echo 
exit 1

