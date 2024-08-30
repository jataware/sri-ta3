#!/bin/bash

# install.sh

conda create -n cuml_env -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.08 python=3.11 'cuda-version>=12.0,<=12.5'
    
pip install pyarrow --upgrade
pip install git+https://github.com/bkj/rcode
pip install imagecodecs
pip install google-vizier[jax]