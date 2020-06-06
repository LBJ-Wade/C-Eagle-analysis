
from mpi4py import MPI
from .__init__ import pprint
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def main():
    from .read import find_files, commune, fof_header, fof_groups, snap_groupnumbers, cluster_particles

    REDSHIFT = 'z003p000'
    N_HALOS = 100

    # -----------------------------------------------------------------------
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fof_groups = fof_groups(files, header)
    # snap_groupnumbers = snap_groupnumbers(files, fof_groups)
    # cluster_particles = cluster_particles(files, groupNumbers=snap_groupnumbers)
    print(fof_groups.keys())




