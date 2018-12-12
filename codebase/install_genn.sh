#! /bin/bash

### requries swig

source ./path_config.sh
source ./install_venv_new.sh

source $VENV_NEW_DIR/bin/activate

./has_nvidia_cuda.sh
HAS_CUDA=$?
echo "HAS_CUDA = $HAS_CUDA"

git clone https://github.com/genn-team/genn $GENN_DIR

cd $GENN_DIR
cd lib
if [ $HAS_CUDA -eq 1 ]
then
    make -f GNUMakefileLibGeNN LIBGENN_PATH=$GENN_DIR/pygenn/genn_wrapper/
    make -f GNUMakefileLibGeNN DYNAMIC=True LIBGENN_PATH=$GENN_DIR/pygenn/genn_wrapper/
else
    make -f GNUMakefileLibGeNN LIBGENN_PATH=$GENN_DIR/pygenn/genn_wrapper/ CPU_ONLY=1
    make -f GNUMakefileLibGeNN DYNAMIC=True LIBGENN_PATH=$GENN_DIR/pygenn/genn_wrapper/ CPU_ONLY=1
fi

cd ..
python setup.py develop


cd ..
git clone https://github.com/genn-team/pynn_genn $PYNN_GENN_DIR

cd $PYNN_GENN_DIR
python setup.py develop

deactivate
