from os import system as cmd
from time.time import sleep

cmd('module unload python/2.7.15')
sleep(1)
cmd('module load python/3.6.5')
sleep(1)
cmd('git pull')
sleep(2)
cmd('python3 main.py')