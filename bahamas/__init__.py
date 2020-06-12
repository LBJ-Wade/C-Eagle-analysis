from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from rotvel_correlation.alignment import save_report
from save.dict2hdf import recursively_save_dict_contents_to_group as save_group