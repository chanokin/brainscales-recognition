#! /bin/bash

source ./path_config.sh

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_NEW_DIR" ]; then
    virtualenv $VENV_NEW_DIR
    source $VENV_NEW_DIR/bin/activate
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# add codebase to the PATH env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export PATH=$BASE_DIR:$PATH" >> "$VENV_NEW_DIR/bin/activate"
    
    echo "" >> "$VENV_NEW_DIR/bin/activate"
    echo "# create the BASE_DIR env variable" >> "$VENV_NEW_DIR/bin/activate"
    echo "export BASE_DIR=$BASE_DIR" >> "$VENV_NEW_DIR/bin/activate"
    $VENV_NEW_DIR/bin/pip install -U pip
    deactivate
fi

# pip install --upgrade pip
