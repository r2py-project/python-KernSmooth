#!/bin/bash
#$ -pe smp 1
#$ -q long
module load gcc
source /opt/crc/c/conda/23.5.2/etc/profile.d/conda.sh
conda activate r-to-python
pip install --no-build-isolation .