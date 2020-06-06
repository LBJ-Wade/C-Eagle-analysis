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
        cluster_particles,
        cluster_data,
        glance_cluster
    )

    REDSHIFT = 'z003p000'
    NHALOS = 10

    # -----------------------------------------------------------------------
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files, header)
    groupnumbers = snap_groupnumbers(files, fofgroups = fofs)

    # for i in range(NHALOS):
    halo_data = cluster_data(NHALOS, files, header, fofgroups = fofs, groupNumbers = groupnumbers)
    pprint('[+] Glance cluster info')
    glance_cluster(halo_data)
    MPI.COMM_WORLD.Barrier()