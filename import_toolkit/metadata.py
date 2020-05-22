import sys
import os
import itertools
import numpy as np
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation

def check_dirs(simulation_obj) -> np.ndarray:
    """
    Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    are present in the simulation archive.
    :return:
    """
    iterator = itertools.product(simulation_obj.clusterIDAllowed, simulation_obj.redshiftAllowed)
    check_matrix = np.zeros((len(simulation_obj.clusterIDAllowed), len(simulation_obj.redshiftAllowed)), dtype=np.bool)
    for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
        cluster = Cluster(simulation_name=simulation_obj.simulation_name,
                          clusterID=halo_id,
                          redshift=halo_z)
        test = cluster.is_cluster() * cluster.is_redshift()
        check_matrix[halo_id][simulation_obj.redshiftAllowed.index(halo_z)] = test
        if not test:
            print(process_n, halo_id, halo_z)
    return check_matrix



def bahamas_mass_cut(cluster):
    n_largeM = 0
    n_total = 0
    for counter, file in enumerate(cluster.groups_filePaths()):
        print(f"[+] Analysing eagle_subfind_tab file {counter}")
        with h5py.File(file, 'r') as group_file:
            m500 = group_file['/FOF/Group_M_Crit500'][:] * 10 ** 10
            n_total += len(m500)
            m_filter = np.where(m500 > 10 ** 13)[0]
            n_largeM += len(m_filter)

# def groupnumber_ghosting(self) -> None:
#     assert hasattr(self, 'ghost')
#     self.ghost.show_yourself()
#     if self.ghost.is_awake(self.redshift):
#         del self.ghost.tagger, self.ghost.memory
#         self.ghost.tagger = self.redshift
#         ghost_mem = dict()
#         print('Loading particle groupNumbers to ghost memory...')
#         for file in self.partdata_filePaths():
#             with h5.File(file, 'r') as h5file:
#                 for key in self.requires:
#                     if key.startswith('partType'):
#                         part_gn = h5file[f'/PartType{key[-1]:s}/GroupNumber'][:]
#                         ghost_mem[f"{key:s}_groupnumber"] = part_gn
#                         del part_gn
#         self.ghost.memory = ghost_mem
#         self.ghost.show_yourself()
