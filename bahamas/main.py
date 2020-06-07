import warnings
import datetime
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
    halo_load_time = []
    for i in range(NHALOS):
        start = datetime.datetime.now()
        halo_data = cluster_data(i, files, header, fofgroups = fofs, groupNumbers = part_gn)
        pprint('[+] Glance cluster info:')
        glance_cluster(halo_data)
        del halo_data
        end = datetime.datetime.now()
        # Time it
        halo_load_time.append((end - start).total_seconds())
        del start, end
        if NHALOS < 5 or len(halo_load_time) < 5:
            completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        else:
            completion_time = sum(halo_load_time[-4:]) / 4 * NHALOS
        pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")

    MPI.COMM_WORLD.Barrier()

