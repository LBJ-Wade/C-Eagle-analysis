#!/usr/bin/env bash
> access.log
#/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 12 python -u ./main.py > ./main.log &
/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 3 python3  ./main.py