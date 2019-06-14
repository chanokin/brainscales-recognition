#! /bin/bash

source ./path_config.sh

function appendToConfig(){
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo $1 >> "$VENV_NEW_DIR/bin/activate"
    echo $2 >> "$VENV_NEW_DIR/bin/activate"
}

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_NEW_DIR" ]; then
    
    python3 -m venv $VENV_NEW_DIR
    source $VENV_NEW_DIR/bin/activate

    pip install --upgrade pip
    
    appendToConfig "# add codebase to the PATH env variable" \
                   "export PATH=$BASE_DIR:\$PATH"
    
    appendToConfig "# create the BASE_DIR env variable" \
                   "export BASE_DIR=$BASE_DIR"

    appendToConfig  "# create the CUDA_PATH env variable" \
                    "export CUDA_PATH=$CUDA_DIR"

    appendToConfig "# append CUDA lib path" \
                   "export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"

    appendToConfig "# append CUDA bin to PATH" \
                   "export PATH=\$CUDA_PATH/bin:\$PATH"

    appendToConfig "# create the GENN_PATH env variable" \
                   "export GENN_PATH=$GENN_DIR"
    
    appendToConfig "# append GENN's binary path to the PATH env variable" \
                   "export PATH=\$GENN_PATH/lib/bin:\$PATH"


    $VENV_NEW_DIR/bin/pip install numpy
    $VENV_NEW_DIR/bin/pip install scipy
    $VENV_NEW_DIR/bin/pip install matplotlib
    $VENV_NEW_DIR/bin/pip install Pillow
    $VENV_NEW_DIR/bin/pip install Pillow-PIL
    $VENV_NEW_DIR/bin/pip install dill
    # $VENV_NEW_DIR/bin/pip install pyro4
    $VENV_NEW_DIR/bin/pip install deap
    $VENV_NEW_DIR/bin/pip install pypet

    deactivate
else
    source $VENV_NEW_DIR/bin/activate
fi
