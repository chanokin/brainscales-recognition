#! /bin/bash
source ./path_config.sh
source ./install_venv_new.sh

### dependencies
### libboost-dev libeigen3-dev  libnlopt0 libnlopt-dev coinor-libipopt1v5 coinor-libipopt-dev

### download 
git clone https://github.com/esa/pagmo2.git pagmo2_install
cd pagmo2_install 
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$PAGMO_DIR \
      -DPAGMO_WITH_EIGEN3=ON \
      -DPAGMO_WITH_NLOPT=ON \
      -DPAGMO_WITH_IPOPT=ON \
      ..

### compile
make

### install
make install

### clean up
# make clean
cd ../..
rm -fr pagmo2_install

