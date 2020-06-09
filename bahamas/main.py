import warnings
import datetime
from mpi4py import MPI
warnings.filterwarnings("ignore")

def main():
    from .__init__ import pprint
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
    from .utils import (
        file_benchmarks,
        display_benchmarks,
        record_benchmarks,
        time_checkpoint
    )

    # Upper level relative imports
    from .__init__ import Cluster, save_alignment_report

    REDSHIFT = 'z003p000'
    HALOSTART = 0
    NHALOS = 5#14366

    # -----------------------------------------------------------------------
    # Initialise benckmarks
    file_benchmarks(REDSHIFT)
    halo_load_time = []
    # -----------------------------------------------------------------------


    # Load snapshot data
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files)
    snap_partgn = snap_groupnumbers(fofgroups = fofs)

    for i in range(HALOSTART, HALOSTART+NHALOS+1, 1):
        start = datetime.datetime.now()

        halo_data = cluster_data(i, header, fofgroups=fofs, groupNumbers=snap_partgn)

        record_benchmarks(REDSHIFT, ('load', i, time_checkpoint(start)))

        pprint('[+] Glance cluster info:')
        glance_cluster(halo_data)
        start = datetime.datetime.now()

        cluster = Cluster.from_dict(simulation_name='bahamas', data=halo_data)
        del halo_data
        align_report = save_alignment_report(cluster)

        record_benchmarks(REDSHIFT, ('compute', i, time_checkpoint(start)))








        # -----------------------------------------------------------------------
        # Time it
        halo_load_time.append((datetime.datetime.now() - start).total_seconds())
        if NHALOS < 5 or len(halo_load_time) < 5:
            completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        else:
            completion_time = sum(halo_load_time[-4:]) / 4 * (HALOSTART+NHALOS-i)
        pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")

    MPI.COMM_WORLD.Barrier()
    display_benchmarks(REDSHIFT)

