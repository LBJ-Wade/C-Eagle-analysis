import itertools
import numpy as np
from simulation import Simulation
from cluster import Cluster

def check_dirs(self) -> np.ndarray:
    """
    Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    are present in the simulation archive.
    :return:
    """
    iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
    check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
    for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
        c = Cluster(simulation_name=self.simulation_name,
                          clusterID=halo_id,
                          redshift=halo_z)

        test = c.is_cluster() * c.is_redshift()
        check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test

        if not test:
            print(process_n, halo_id, halo_z)

    return check_matrix


sim = Simulation(simulation_name='ceagle')
check_dirs(sim)


def bahamas_mass_cut():
    n_largeM = 0
    n_total = 0
    cluster = Cluster(simulation_name='bahamas',
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