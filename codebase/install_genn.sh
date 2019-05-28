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
make


if [ $HAS_CUDA -eq 1 ]
then
    make LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/
    make DYNAMIC=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/
    make DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/
    make MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/
fi

make CPU_ONLY=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/ 
make CPU_ONLY=True DYNAMIC=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/ 
make CPU_ONLY=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/
make CPU_ONLY=True DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN_DIR/pygenn/genn_wrapper/



python setup.py develop


cd ..
git clone -b genn_4 https://github.com/genn-team/pynn_genn $PYNN_GENN_DIR

cd $PYNN_GENN_DIR
python setup.py develop

deactivate
