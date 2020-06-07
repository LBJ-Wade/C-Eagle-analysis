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
    from .utils import (
        file_benchmarks,
        display_benchmarks
    )

    REDSHIFT = 'z003p000'
    HALOSTART = 25
    NHALOS = 300

    # -----------------------------------------------------------------------
    # Initialise benckmarks
    timing_filename = file_benchmarks(REDSHIFT)
    display_benchmarks(REDSHIFT)
    halo_load_time = []

    # Load snapshot data
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files)
    snap_partgn = snap_groupnumbers(fofgroups = fofs)

    for i in range(HALOSTART, HALOSTART+NHALOS, 1):
        start = datetime.datetime.now()
        halo_data = cluster_data(i, header, fofgroups = fofs, groupNumbers = snap_partgn)
        end = datetime.datetime.now()
        pprint('[+] Glance cluster info:')
        glance_cluster(halo_data)
        del halo_data

        # Time it
        elapsed = (end - start).total_seconds()
        halo_load_time.append(elapsed)
        del start, end
        if NHALOS < 5 or len(halo_load_time) < 5:
            completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        else:
            completion_time = sum(halo_load_time[-4:]) / 4 * NHALOS
        pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")

        # Print benckmarks to file
        with open(timing_filename, "a") as benchmarks:
            pprint(f"{i},{elapsed}", file=benchmarks)

    MPI.COMM_WORLD.Barrier()
    display_benchmarks(REDSHIFT)

