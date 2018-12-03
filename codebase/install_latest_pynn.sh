#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh
source $NEST_NEW_DIR/bin/nest_vars.sh

### requires mpi-default-dev

### install PyNN - latest
pip install numpy
pip install scipy
pip install matplotlib
pip install six
pip install lazyarray
pip install neo
pip install mpi4py
pip install jinja2
pip install csa
pip install PyNN
