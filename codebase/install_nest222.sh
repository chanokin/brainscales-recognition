#! /bin/bash
source ./path_config.sh
source $VENV_DIR/bin/activate

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
rm -fr nest_install

# add nest Python paths
echo "$NEST_DIR/lib/python2.7/site-packages/" > $VENV_DIR/lib/python2.7/site-packages/nest.pth
