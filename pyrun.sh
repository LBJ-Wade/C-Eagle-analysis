module unload python/2.7.15
module load python/3.6.5
python3 --version
echo -e "\e[1m \e[92m Syncing GitHub..."
git pull
echo Running main program...
python3 main.py
echo End of session.