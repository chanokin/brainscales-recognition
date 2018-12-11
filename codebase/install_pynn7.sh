#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh
source $VENV_NEW_DIR/bin/activate

# install PyNN 0.7.5
pip install PyNN==0.7.5 --target=$PYNN7_DIR --no-deps

deactivate
