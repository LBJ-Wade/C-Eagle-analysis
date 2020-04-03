import itertools
import numpy as np
from . import simulation
from . import cluster

def check_dirs(self) -> np.ndarray:
    """
    Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    are present in the simulation archive.
    :return:
    """
    iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
    check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
    for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
        c = cluster.Cluster(simulation_name=self.simulation_name,
                          clusterID=halo_id,
                          redshift=halo_z)

        test = c.is_cluster() * c.is_redshift()
        check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test

        if not test:
            print(process_n, halo_id, halo_z)

    return check_matrix


if __name__ == '__main__':
    sim = simulation.Simulation(simulation_name='ceagle')
    sim.check_dirs(sim)