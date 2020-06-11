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
        time_checkpoint
    )

    # Upper level relative imports
    from .__init__ import Cluster, save_report, write

    REDSHIFT = 'z001p000'
    HALOSTART = 0
    NHALOS = 14366

    # -----------------------------------------------------------------------
    # Initialise benchmarks
    file_benchmarks(REDSHIFT)
    halo_load_time = []
    # -----------------------------------------------------------------------
    # Initialise snapshot output file
    snap_output = {}
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

        start = datetime.datetime.now()
        cluster = Cluster.from_dict(simulation_name='bahamas', data=halo_data)
        del halo_data
        snap_output[f"halo_{i:d}"] = save_report(cluster)
        record_benchmarks(REDSHIFT, ('compute', i, time_checkpoint(start)))

        # -----------------------------------------------------------------------
        # Time it
        halo_load_time.append((datetime.datetime.now() - start).total_seconds())
        if NHALOS < 5 or len(halo_load_time) < 5:
            completion_time = sum(halo_load_time)/len(halo_load_time) * NHALOS
        else:
            completion_time = sum(halo_load_time[-4:]) / 4 * (HALOSTART+NHALOS-i)
        pprint(f"[x] ({len(halo_load_time):d}/{NHALOS:d}) Estimated completion time: {datetime.timedelta(seconds=completion_time)}")
        # -----------------------------------------------------------------------

    MPI.COMM_WORLD.Barrier()
    # Save snapshot resutls
    pathFile = os.path.join(cluster.pathSave, 'alignment_project')
    if not os.path.exists(pathFile):
        os.makedirs(pathFile)
    write.save_dict_to_hdf5(snap_output, os.path.join(pathFile, f"snap_alignment_{REDSHIFT}.hdf5"))

    display_benchmarks(REDSHIFT)

