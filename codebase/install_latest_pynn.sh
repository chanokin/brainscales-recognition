#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh

source $VENV_NEW_DIR/bin/activate

### requires mpi-default-dev


### install PyNN - latest
pip install wheel
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

# pip install PyNN==0.7.5 --target="$PYNN7_DIR"
# touch $PYNN7_DIR/__init__.py

# git clone https://github.com/HumanBrainProject/hbp-neuromorphic-client
# cd hbp-neuromorphic-client
# python setup.py install --user

deactivate
