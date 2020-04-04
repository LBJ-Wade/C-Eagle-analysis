#!/bin/bash -l

#SBATCH --ntasks 224                                           # The number of cores you need...
#SBATCH -J celrs                                            # Give it something meaningful.
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma6                                              # or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004                                               # e.g. dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mail-type=END                                        # notifications for job done & fail
#SBATCH --mail-user=edoardo.altamura@manchester.ac.uk        #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#load the modules used to build your program.
#module unload python/2.7.15
module load python/3.6.5

module load intel_comp/2018
module load openmpi
module load hdf5

# Run the program
mpiexec -n $SLURM_NTASKS python3 ./save/fof_output.py



### Multiples of 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608
### Multiples of 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308, 336, 364, 392, 420, 448, 476, 504, 532