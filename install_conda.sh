#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=multiphys

conda create -n $CONDA_ENV_NAME python=3.9 -y

conda activate $CONDA_ENV_NAME

# install pytorch using pip, update with appropriate cuda drivers if necessary
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117
# uncomment if pip installation isn't working
# conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y


# install remaining requirements
pip install -r requirements.txt