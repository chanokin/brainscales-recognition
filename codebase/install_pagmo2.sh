#! /bin/bash
source ./path_config.sh
source $VENV_DIR/bin/activate

# dependencies
# libboost-dev libeigen3-dev  libnlopt0 libnlopt-dev coinor-libipopt1v5 coinor-libipopt-dev

# download 
git clone https://github.com/esa/pagmo2.git pagmo2_install
cd pagmo2_install 
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$PAGMO_DIR -DPAGMO_WITH_EIGEN3=ON -DPAGMO_WITH_NLOPT=ON -DPAGMO_WITH_IPOPT=ON ..

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
