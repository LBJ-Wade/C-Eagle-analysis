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
#n = "$(nproc)"
n = "$(4)"
echo -e "\e[1m\e[91mRunning main program on "${n}"...\e[0m"
echo -e "+-----------------------------------------------------------------------------------------+"

# Run python in parallel MPI and in background
#mpiexec -n "${n}" python3 -u ./main.py > ./main.log &

# Run python in parallel MPI online
#mpiexec -n "${n}" python3 ./main.py

# Run python on single core online
python3 ./main.py

echo -e "\e[5m\e[1m\e[91mEnd of session.\e[0m"