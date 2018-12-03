#! /bin/bash

source ./path_config.sh

# create virtual environment (Python 2.7)
if [ ! -d "$VENV_DIR" ]; then
    virtualenv $VENV_DIR
    echo "export PATH=$BASE_DIR:$PATH" >> "$VENV_DIR/bin/activate"
fi

source $VENV_DIR/bin/activate
