#!/usr/bin/env bash
> access.log
/usr/local/bin/old-mpi/mpiexec -n 12 python -u ~/C-Eagle-analysis/main.py > ~/C-Eagle-analysis/main.log &