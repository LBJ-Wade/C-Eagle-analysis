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

    REDSHIFT = 'z000p000'
    NHALOS = 10

    # -----------------------------------------------------------------------
    # Load snapshot data
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files, header)
    part_gn = snap_groupnumbers(files, fofgroups = fofs)

    for i in range(NHALOS):
        halo_data = cluster_data(i, files, header, fofgroups = fofs, groupNumbers = part_gn)
        pprint('[+] Glance cluster info:')
        glance_cluster(halo_data)
        del halo_data
    MPI.COMM_WORLD.Barrier()