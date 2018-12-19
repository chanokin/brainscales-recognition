#! /bin/bash
source ./path_config.sh
source ./install_venv_new.sh
source $VENV_NEW_DIR/bin/activate

### install the (many) requirements for pypet 0.3.0
### why is this not in the setup file?
pip install -r pypet0p3p0_reqs.txt
#their own brew of accessible dictionary
pip install https://github.com/IGITUGraz/sdict/archive/master.zip

### requires libyaml-dev

### finally, install pypet
git clone https://github.com/IGITUGraz/L2L $L2L_DIR

cd $L2L_DIR
python setup.py develop
