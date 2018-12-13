#! /bin/bash

source ./path_config.sh
source ./install_venv_new.sh

source $VENV_NEW_DIR/bin/activate

### dependends on zlibc zlib1g-dev libboost-all-dev

pip install boost

mkdir $BSS_STACK_DIR
cd $BSS_STACK_DIR

# WAF is an automated build system/toolset
git clone https://gitlab.com/ita1024/waf
cd waf
./waf-light --tools=boost configure install
# ./configure
# make
# make install

echo "" >> $VENV_NEW_DIR/bin/activate
echo "# add waf lib path to environment" >> $VENV_NEW_DIR/bin/activate
echo "export BSS_STACK_DIR=$BSS_STACK_DIR" >> $VENV_NEW_DIR/bin/activate
echo "export WAFDIR=\$BSS_STACK_DIR/waf/waflib" >> $VENV_NEW_DIR/bin/activate

echo "" >> $VENV_NEW_DIR/bin/activate
echo "# add waf bin path to environment" >> $VENV_NEW_DIR/bin/activate
echo "export PATH=\$BSS_STACK_DIR/waf:\$PATH" >> $VENV_NEW_DIR/bin/activate

echo "" >> $VENV_NEW_DIR/bin/activate
echo "# add waf extras to python in environment" >> $VENV_NEW_DIR/bin/activate
echo "export PYTHONPATH=\$WAFDIR/extras/:\$PYTHONPATH" >> $VENV_NEW_DIR/bin/activate
cd ..

WAF=$BSS_STACK_DIR/waf/waf

### activate new environment variables
deactivate
source $VENV_NEW_DIR/bin/activate


### dependency steps
### lib-rcf
git clone https://github.com/electronicvisions/lib-rcf
cd lib-rcf
$WAF configure 
$WAF install -v
cd ..


### hicann-system
git clone https://github.com/electronicvisions/hicann-system
cd hicann-system
$WAF configure 
$WAF install -v
cd ..

### hardware abstraction layer
 git clone https://github.com/electronicvisions/halbe
$WAF configure 
$WAF install -v
cd pyhalbe
$WAF configure 
$WAF install -v
cd ../..

### pynn compatible interface
https://github.com/electronicvisions/pyhmf

deactivate
