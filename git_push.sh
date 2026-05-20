#!/bin/bash
#$ -pe smp 1
#$ -q long
module load git
export PATH=~/bin:$PATH
# git remove add python-KernSmooth https://github.com/caiyufei8/python-KernSmooth.git
git push origin main
# git remote add r2py_kernsmooth https://github.com/caiyufei8/r2py_kernsmooth.git
git subtree push --prefix=r2py_kernsmooth r2py_kernsmooth main