#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh

### ----------------------------------------------------------------- #
###
### Cypress --- it may be useful for parallel/distributed evaluation of
### individuals in a population
###
### Unfortunatelly, it requires PyNN 0.8 and up for Nest! This is not 
### compatible with BrainScaleS?
###
### ----------------------------------------------------------------- #

### required for Cypress - C++ PyNN wrapper
pip install requests
pip install pyminifier
pip install pybind
pip install pybind11

### get Cypress 
git clone https://github.com/hbp-sanncs/cypress tmp_cypress
git clone https://github.com/pybind/pybind11 tmp_cypress/external/pybind11

### patch bug on CMakeLists.txt
sed -i \
's/{CMAKE_BINARY_DIR}\/cypress\/config.h/{CMAKE_BINARY_DIR}\/include\/cypress\/config.h/g' \
tmp_cypress/CMakeLists.txt

### build
mkdir tmp_cypress/build
cd tmp_cypress/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$CYPRESS_DIR ..
echo "--------------------------------------------------\n"
echo "--------------- done configuring -----------------\n"

make
echo "--------------------------------------------------\n"
echo "------------------ done building -----------------\n"

# make test
# echo "--------------------------------------------------\n"
# echo "------------------ done testing ------------------\n"

### install
make install
echo "--------------------------------------------------\n"
echo "---------------- done installing -----------------\n"


### clean up
make clean
echo "--------------------------------------------------\n"
echo "------------------ done cleaning -----------------\n"

cd ../..
rm -fr tmp_cypress
