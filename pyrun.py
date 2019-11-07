from os import system as cmd

cmd('module unload python/2.7.15')
cmd('module load python/3.6.5')
cmd('git pull')
cmd('python3 main.py')