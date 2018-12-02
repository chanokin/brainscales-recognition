#! /bin/bash

source ./path_config.sh
source $VENV_DIR/bin/activate

# ------------------------------------------------------------------- #
#
# Cypress --- it may be useful for parallel/distributed evaluation of
# individuals in a population
#
# Unfortunatelly, it requires PyNN 0.8 and up for Nest! This is not 
# compatible with BrainScaleS?
#
# ------------------------------------------------------------------- #

# required for Cypress - C++ PyNN wrapper
pip install requests
pip install pyminifier
pip install pybind
pip install pybind11

# get Cypress 
git clone https://github.com/hbp-sanncs/cypress tmp_cypress
git clone https://github.com/pybind/pybind11 tmp_cypress/external/pybind11

# build
mkdir tmp_cypress/build
cd tmp_cypress/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$CYPRESS_DIR
make

# install
make install

# clean up
rm -fr tmp_cypress
