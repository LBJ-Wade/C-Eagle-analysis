module unload python/2.7.15
module load python/3.6.5
python3 --version
echo -e "\e[91mSyncing GitHub...\e[0m"
git pull
echo -e "\e[1m\e[91mRunning main program...\e[0m\e[92m"
python3 main.py
echo -e "\e[0m\e[5m\e[1m\e[91mEnd of session.\e[0m"