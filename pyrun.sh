#!/usr/bin/env bash
# Load python 3 in the module packages
echo -e "\e[91mLoading python3...\e[0m"
module unload python/2.7.15
module load python/3.6.5
python3 --version

# Load MPI tools
echo -e "\e[91mLoading MPI...\e[0m"
module load intel_comp/2018
module load openmpi

# Sync GitHub repository
echo -e "\e[91mSyncing GitHub...\e[0m"
git pull
date

# Run main program

echo -e "\e[1m\e[91mRunning main program...\e[0m"
echo -e "+-----------------------------------------------------------------------------------------+"

# Run python in parallel MPI and in background
#mpiexec -n 4 python3 -u ./main.py > ./main.log &

# Run python in parallel MPI online
#mpiexec -n 30 python3 ./main.py

# Run python on single core online
python3 ./main.py

echo -e "\e[5m\e[1m\e[91mEnd of session.\e[0m"