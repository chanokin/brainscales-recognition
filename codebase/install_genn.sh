#! /bin/bash

### requries swig

source ./path_config.sh
source ./install_venv_new.sh

source $VENV_NEW_DIR/bin/activate

git clone https://github.com/genn-team/genn $GENN_DIR

# cd $GENN_DIR
# python setup.py develop


# cd ..
# git clone https://github.com/genn-team/pynn_genn $PYNN_GENN_DIR

# cd $PYNN_GENN_DIR
# python setup.py develop

deactivate
