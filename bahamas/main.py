import os
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
        time_checkpoint,
        report_file,
        error_file
    )

    # Upper level relative imports
    from .__init__ import rank, Cluster, save_report, save_group

    # -----------------------------------------------------------------------
    # Set simulation parameters
    REDSHIFT = 'z001p750'
    HALOSTART = 0
    HALOEND = 14365


    # -----------------------------------------------------------------------
    # Initialise benchmarks
    file_benchmarks(REDSHIFT)
    halo_load_time = []


    # -----------------------------------------------------------------------
    # Initialise snapshot output file
    snap_file = report_file(REDSHIFT)
    error_id = []


    # -----------------------------------------------------------------------
    # Load snapshot data
    pprint('[+] BAHAMAS HYDRO')
    files = find_files(REDSHIFT)
    header = fof_header(files)
    fofs = fof_groups(files)
    snap_partgn = snap_groupnumbers(fofgroups = fofs)

    for i in range(HALOSTART, HALOEND+1, 1):

        # Extract data from subfind output
        start_1 = datetime.datetime.now()
        halo_data = cluster_data(i, header, fofgroups=fofs, groupNumbers=snap_partgn)
        record_benchmarks(REDSHIFT, ('load', i, time_checkpoint(start_1)))

        # Parse data into Cluster object
        start_2 = datetime.datetime.now()
        cluster = Cluster.from_dict(simulation_name='bahamas', data=halo_data)
        del halo_data

        # Try pushing results into an h5 file
        try:
            halo_report = save_report(cluster)
            del cluster
        except:
            error_id.append(i)
        else:
            if rank == 0: save_group(snap_file, f"/halo_{i:05d}/", halo_report)
            del halo_report

        record_benchmarks(REDSHIFT, ('compute', i, time_checkpoint(start_2)))

        # -----------------------------------------------------------------------
        # Time it
        # halo_load_time.append((datetime.datetime.now() - start_1).total_seconds())
        # if NHALOS < 5 or len(halo_load_time) < 5:
        #     completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        # else:
        #     completion_time = sum(halo_load_time[-4:]) / 4 * (HALOSTART+NHALOS-i+1)
        # pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")
        # -----------------------------------------------------------------------

    MPI.COMM_WORLD.Barrier()
    if rank==0: snap_file.close()

    # Save IDs of halos with errors
    error_file(REDSHIFT, error_id)
    display_benchmarks(REDSHIFT)

