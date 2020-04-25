import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size is 3

if rank is 0:
	os.system("python3 ./simstats.py -s celr_e")
elif rank is 1:
	os.system("python3 ./simstats.py -s celr_b")
elif rank is 2:
	os.system("python3 ./simstats.py -s macsis")
