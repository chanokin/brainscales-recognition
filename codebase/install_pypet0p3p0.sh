#! /bin/bash
source ./path_config.sh
source ./install_venv_new.sh
source $VENV_NEW_DIR/bin/activate

### install the (many) requirements for pypet 0.3.0
### why is this not in the setup file?
pip install -r pypet0p3p0_reqs.txt

### finally, install pypet
pip install --no-deps pypet==0.3.0 
