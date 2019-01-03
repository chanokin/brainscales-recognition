#! /bin/bash

source ../path_config.sh
echo "installing new virtual envirnoment in "
echo $VENV_NEW_DIR

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_NEW_DIR" ]; then
    python3 -m venv $VENV_NEW_DIR
    source $VENV_NEW_DIR/bin/activate
    pip install -U pip
    
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the BASE_DIR env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export BASE_DIR=$BASE_DIR" >> "$VENV_NEW_DIR/bin/activate"

    $VENV_NEW_DIR/bin/pip install numpy
    $VENV_NEW_DIR/bin/pip install scipy
    $VENV_NEW_DIR/bin/pip install matplotlib
    $VENV_NEW_DIR/bin/pip install Pillow
    $VENV_NEW_DIR/bin/pip install Pillow-PIL
    $VENV_NEW_DIR/bin/pip install pyro4
    deactivate
else
    source $VENV_NEW_DIR/bin/activate
fi
