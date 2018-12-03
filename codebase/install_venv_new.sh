#! /bin/bash

source ./path_config.sh

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_NEW_DIR" ]; then
    virtualenv $VENV_NEW_DIR
    echo "export PATH=$BASE_DIR:$PATH" >> "$VENV_NEW_DIR/bin/activate"
    echo "export BASE_DIR=$BASE_DIR" >> "$VENV_NEW_DIR/bin/activate"
fi

source $VENV_NEW_DIR/bin/activate
