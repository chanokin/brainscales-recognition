#! /bin/bash
source ./path_config.sh
source ./install_venv2.sh

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
echo "--------------------------------------------------\n"
echo "--------------------------------------------------\n"

make install

echo "--------------------------------------------------\n"
echo "------------------- done install -----------------\n"


# clean up
make clean
echo "--------------------------------------------------\n"
echo "-------------------  done clean  -----------------\n"

cd ../..
rm -fr nest_install
echo "--------------------------------------------------\n"
echo "------------------ done removing -----------------\n"

# add nest Python paths
echo "$NEST_DIR/lib/python2.7/site-packages/" > $VENV_DIR/lib/python2.7/site-packages/nest.pth
