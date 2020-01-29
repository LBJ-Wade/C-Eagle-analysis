#!/bin/bash -l

#SBATCH --ntasks 143                                           # The number of cores you need...
#SBATCH -J CELR_red                                            # Give it something meaningful.
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma6                                              # or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004                                               # e.g. dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mail-type=END                                        # notifications for job done & fail
#SBATCH --mail-user=<edoardo.altamura@manchester.ac.uk>        #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
module unload python/2.7.15
module load python/3.6.5

module load intel_comp/2018
module load openmpi

# Run the program
mpiexec -n $SLURM_NTASKS python3 ./main.py