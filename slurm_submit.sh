#!/bin/bash -l

#SBATCH --ntasks 144                                           # The number of cores you need...
#SBATCH -J CELR_stage1                                         # Give it something meaningful.
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma-analyse                                       # or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004                                               # e.g. dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mail-type=CELR_stage1-END                            # notifications for job done & fail
#SBATCH --mail-user=<edoardo.altamura@manchester.ac.uk>        #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module unload python/2.7.15
module load python/3.6.5

module load intel_comp
module load intel_mpi
module load hdf5


# Run the program
mpirun -np $SLURM_NTASKS python3 /cosma/home/dp004/dc-alta2/C-Eagle-analysis/main.py