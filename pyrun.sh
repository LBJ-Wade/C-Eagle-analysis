# Load python 3 in the module packages
module unload python/2.7.15
module load python/3.6.5
python3 --version

# Sync GitHub repository
echo -e "\e[91mSyncing GitHub...\e[0m\e[93"
git pull
date

# Run main program
echo -e "\e[1m\e[91mRunning main program..."
echo -e "-------------------------------------------------------------------\e[0m"

#python3 -u ./main.py > ./main.log &
python3 ./main.py

echo -e "\e[5m\e[1m\e[91mEnd of session.\e[0m"