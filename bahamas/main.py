import warnings
from mpi4py import MPI
from .__init__ import pprint
warnings.filterwarnings("ignore")

def main():
    from .read import (
        find_files,
        fof_header,
        fof_groups,
        fof_group,
        snap_groupnumbers,
        cluster_particles
    )

    REDSHIFT = 'z003p000'
    NHALOS = 10

    # -----------------------------------------------------------------------
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fof_groups = fof_groups(files, header)
    snap_groupnumbers = snap_groupnumbers(files, fofgroups = fof_groups)

    for i in range(NHALOS):
        fof_group = fof_group(i, fofgroups = fof_groups)
        pprint(fof_group)
        cluster_particles = cluster_particles(files, header, fofgroup=fof_group, groupNumbers=snap_groupnumbers)
        pprint(cluster_particles)
    MPI.COMM_WORLD.Barrier()




