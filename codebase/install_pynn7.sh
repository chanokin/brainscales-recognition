#! /bin/bash

source ./path_config.sh

# create virtual environment (Python 2.7)
virtualenv $VENV_DIR
source $VENV_DIR/bin/activate

# install PyNN 0.7.5
pip install numpy
pip install six
pip install scipy
pip install matplotlib
pip install PyNN==0.7.5
