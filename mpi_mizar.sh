#!/usr/bin/env bash
git pull
> access.log
/usr/local/bin/old-mpi/mpiexec -n 12 python3 -u ./main.py > ./main.log &