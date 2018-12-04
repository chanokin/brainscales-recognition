#! /bin/bash
source ./path_config.sh
source ./install_venv_new.sh
source $VENV_NEW_DIR/bin/activate

# requires cython for pynest
pip install cython 

mkdir neurosim_install
cd neurosim_install
git clone https://github.com/INCF/libneurosim
cd libneurosim
./autogen.sh
./configure --prefix=$NEUROSIM_DIR
make
make install
make clean
cd ../..
rm -fr neurosim_install

### create temporary install dir
mkdir nest_install
cd nest_install
git clone https://github.com/nest/nest-simulator


### configure
###   requires: libncurses5-dev libncurses5 libncurses5-dev libncurses5
###   optional: libgsl2 libgsl-dev
cd nest-simulator

mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_NEW_DIR  \
      -Dwith-mpi=ON \
      -Dwith-libneurosim=$NEUROSIM_DIR/lib \
      ..

### compile
make

### install
echo "--------------------------------------------------\n"
echo "--------------------------------------------------\n"

make install

echo "--------------------------------------------------\n"
echo "------------------- done install -----------------\n"


### clean up
make clean
echo "--------------------------------------------------\n"
echo "-------------------  done clean  -----------------\n"

cd ../../..
rm -fr nest_install
echo "--------------------------------------------------\n"
echo "------------------ done removing -----------------\n"

### add variables to environment
$NEST_NEW_DIR/bin/nest_vars.sh

### add nest Python paths
echo "$NEST_NEW_DIR/lib/python2.7/site-packages/" > $VENV_NEW_DIR/lib/python2.7/site-packages/nest.pth

deactivate
