#!/bin/bash
source ./path_config.sh
source $VENV_NEW_DIR/bin/activate

# killall -s 9 run_pynn_server.sh

export PYRO_SERIALIZERS_ACCEPTED=pickle,dill,serpent,json,marshal
pyro4-ns 2> /dev/null &
pyro4-nsc removematching spikevo 
python -m spikevo.PyNNServer