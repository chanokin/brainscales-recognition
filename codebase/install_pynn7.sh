#! /bin/bash

source ./path_config.sh
source ./install_venv2.sh

# install PyNN 0.7.5
pip install numpy
pip install six
pip install scipy
pip install matplotlib
pip install PyNN==0.7.5
