#! /bin/bash
source ../path_config.sh
source ./install_venv_py3.sh
source $VENV_NEW_DIR/bin/activate

### requires libyaml-dev, python3-dev

### install the (many) requirements for pypet 0.3.0
### why is this not in the setup file?
pip install -r l2l_reqs.txt

### their own brew of accessible dictionary
pip install https://github.com/IGITUGraz/sdict/archive/master.zip


### JUBE benchmarking environment
wget http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=2.2.1 \
    -O JUBE-2.2.1.tar.gz

tar xf JUBE-2.2.1.tar.gz
mv JUBE-2.2.1 $JUBE_DIR
cd $JUBE_DIR
python setup.py develop
cd ..

echo "$JUBE_DIR" > $VENV_NEW_DIR/lib/python3.5/site-packages/jube.pth


### finally, install learning-to-learn
git clone https://github.com/IGITUGraz/L2L $L2L_DIR

cd $L2L_DIR
python setup.py develop
