"""
This code is auxiliary to any MPI operation involving the mpiexec command.
NB need to assign one thread for every process.
"""

import os
import time

num_threads = 10
py_file_name = 'map_tSZ_intensity.py'

command = 'mpiexec -n '+ str(num_threads)+' python '+ str(py_file_name)

start = time.time()
os.system(command)
end = time.time()
print('Total runtime (s):\t', end-start)