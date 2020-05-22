import sys
import itertools
import numpy as np
import h5py
from . import simulation as s
from . import cluster as c

def check_dirs(self) -> np.ndarray:
    """
    Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    are present in the simulation archive.
    :return:
    """
    iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
    check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
    for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
        cluster = c.Cluster(simulation_name=self.simulation_name,
                          clusterID=halo_id,
                          redshift=halo_z)

        test = cluster.is_cluster() * cluster.is_redshift()
        check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test

        if not test:
            print(process_n, halo_id, halo_z)

    return check_matrix

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def bahamas_mass_cut():
    n_largeM = 0
    n_total = 0
    cluster = c.Cluster(simulation_name='bahamas',
                      clusterID=0,
                      redshift='z000p000',
                      comovingframe=False,
                      fastbrowsing=True)
    for counter, file in enumerate(cluster.groups_filePaths()):
        print(f"[+] Analysing eagle_subfind_tab file {counter}")
        with h5py.File(file, 'r') as group_file:
            m500 = group_file['/FOF/Group_M_Crit500'][:] * 10 ** 10
            n_total += len(m500)
            m_filter = np.where(m500 > 10 ** 13)[0]
            n_largeM += len(m_filter)