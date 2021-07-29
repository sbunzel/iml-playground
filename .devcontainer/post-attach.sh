#!/bin/bash

# Add your commands here
# ------------------------------
source /opt/conda/bin/activate
conda activate devenv
pre-commit install
pip install -e .
