#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aimnet2calc-env

which python
python --version

# Run the aimnet2.py script with all passed arguments.
python aimnet2_flask.py "$@"