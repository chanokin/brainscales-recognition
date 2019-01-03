#! /bin/bash

source ./path_config.sh

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_NEW_DIR" ]; then
    virtualenv $VENV_NEW_DIR
    source $VENV_NEW_DIR/bin/activate
    
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# add codebase to the PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PATH=$BASE_DIR:\$PATH" >> "$VENV_NEW_DIR/bin/activate"
    
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the BASE_DIR env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export BASE_DIR=$BASE_DIR" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the CUDA_PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export CUDA_PATH=$CUDA_DIR" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# append CUDA lib path" >> "$VENV_NEW_DIR/bin/activate"
    echo "export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# append CUDA bin to PATH" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PATH=\$CUDA_PATH/bin:\$PATH" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the GENN_PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export GENN_PATH=$GENN_DIR" >> "$VENV_NEW_DIR/bin/activate"
    
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# append GENN's binary path to the PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PATH=\$GENN_PATH/lib/bin:\$PATH" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the PYNN7_PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PYNN7_PATH=$PYNN7_DIR" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the NEST222_PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export NEST222_PATH=$NEST_DIR" >> "$VENV_NEW_DIR/bin/activate"

    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the PYNEST222_PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PYNEST222_PATH=$NEST_DIR/lib/python2.7/site-packages/" >> "$VENV_NEW_DIR/bin/activate"

    echo "$SPIKEVO_DIR" > $VENV_NEW_DIR/lib/python2.7/site-packages/spikevo.pth

    

    $VENV_NEW_DIR/bin/pip install numpy
    $VENV_NEW_DIR/bin/pip install scipy
    $VENV_NEW_DIR/bin/pip install matplotlib
    $VENV_NEW_DIR/bin/pip install Pillow
    $VENV_NEW_DIR/bin/pip install Pillow-PIL
    $VENV_NEW_DIR/bin/pip install dill
    $VENV_NEW_DIR/bin/pip install pyro4
    deactivate
else
    source $VENV_NEW_DIR/bin/activate
fi
