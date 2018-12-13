#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh

source $VENV_NEW_DIR/bin/activate

pip install requests
git clone https://github.com/HumanBrainProject/hbp-neuromorphic-client
cd hbp-neuromorphic-client
python setup.py install

deactivate
