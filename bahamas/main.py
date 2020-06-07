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
        cluster_partgroupnumbers,
        cluster_particles,
        cluster_data,
        glance_cluster
    )

    REDSHIFT = 'z003p000'
    NHALOS = 3

    # -----------------------------------------------------------------------
    # Load snapshot data
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files)
    snap_partgn = snap_groupnumbers(fofgroups = fofs)
    halo_load_time = []
    for i in range(NHALOS):

        start = datetime.datetime.now()
        fof = fof_group(i, fofgroups = fofs)
        pprint('[+] Glance cluster info:')
        glance_cluster(fof)

        halo_partgn = cluster_partgroupnumbers(fofgroup = fof, groupNumbers = snap_partgn)
        halo_data = cluster_particles(fofgroup = fof, groupNumbers = halo_partgn)
        pprint('[+] Glance cluster info:')
        glance_cluster(halo_data)
        del halo_data

        # Time it
        end = datetime.datetime.now()
        halo_load_time.append((end - start).total_seconds())
        del start, end
        if NHALOS < 5 or len(halo_load_time) < 5:
            completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        else:
            completion_time = sum(halo_load_time[-4:]) / 4 * NHALOS
        pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")

    MPI.COMM_WORLD.Barrier()

