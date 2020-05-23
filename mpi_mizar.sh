#!/usr/bin/env bash
> access.log
/usr/local/openmpi-2.0.1-intel/bin/mpiexec -n 24 python -u ./main.py > ./main.log &