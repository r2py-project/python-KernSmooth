#!/bin/bash
#$ -pe smp 1
#$ -q long
# conda create -n r-to-python python r-base -c conda-forge -y
module load gcc
conda activate r-to-python
conda install -c conda-forge meson-python numpy pandas scipy scikit-learn matplotlib r-kernsmooth -y
