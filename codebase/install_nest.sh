#! /bin/bash

BASE_DIR=`pwd`
VENV_DIR="$BASE_DIR/venv2"
NEST_DIR="$BASE_DIR/nest"
CYPR_DIR="$BASE_DIR/cypress"

# create virtual environment (Python 2.7)
virtualenv $VENV_DIR

# install PyNN 0.7.5
source "$VENV_DIR/bin/activate"
pip install numpy
pip install six
pip install scipy
pip install matplotlib
pip install PyNN==0.7.5

# create temporary install dir
mkdir nest_install
cd nest_install
wget https://github.com/nest/nest-releases/blob/master/nest-2.2.2.tar.gz?raw=true
mv nest-2.2.2.tar.gz?raw=true nest-2.2.2.tar.gz
tar xf nest-2.2.2.tar.gz

# configure
#   requires: libncurses5-dev libncurses5 libncurses5-dev libncurses5
#   optional: libgsl2 libgsl-dev
cd nest-2.2.2 
./configure --prefix=$NEST_DIR

# compile
make

# install
make install

# clean up
make clean
cd ../..
rm -fr nest_install

# add nest Python paths
echo "$INSTALL_DIR/lib/python2.7/site-packages/" > "$VENV_DIR/lib/python2.7/site-packages/nest.pth"

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
# pip install requests
# pip install pyminifier
# pip install pybind
# pip install pybind11

# get Cypress 
# git clone https://github.com/hbp-sanncs/cypress $CYPR_DIR
# git clone https://github.com/pybind/pybind11 $CYPR_DIR/external/pybind11

# build
# mkdir $CYPR_DIR/build
# cd $CYPR_DIR/build
# cmake ..
# make && make test

# install
# make install
