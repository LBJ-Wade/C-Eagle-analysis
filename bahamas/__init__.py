from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)