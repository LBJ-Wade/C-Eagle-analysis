import warnings
from mpi4py import MPI
from .__init__ import pprint
warnings.filterwarnings("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def main():
    from .read import (
        find_files,
        commune,
        fof_header,
        fof_groups,
        fof_group,
        snap_groupnumbers,
        cluster_particles
    )

    REDSHIFT = 'z003p000'
    CLUSTERID = 100

    # -----------------------------------------------------------------------
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fof_groups = fof_groups(files, header)
    snap_groupnumbers = snap_groupnumbers(files, fofgroups = fof_groups)
    fof_group = fof_group(CLUSTERID, fofgroups = fof_groups)
    pprint(fof_group)
    cluster_particles = cluster_particles(files, header, fofgroup=fof_group, groupNumbers=snap_groupnumbers)
    pprint(cluster_particles.keys())
    comm.Barrier()




