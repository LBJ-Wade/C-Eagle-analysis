import sys
import os
import itertools
import numpy as np
import h5py
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.cluster import Cluster

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


def ceagle2yaml() -> None:
    documents = {}
    documents['simulation_name'] = 'ceagle'
    documents['setup'] = 'zoom'
    documents['simulation'] = 'C-EAGLE'
    documents['computer'] = 'cosma5'
    documents['pathData'] = '/cosma5/data/dp004/C-EAGLE/Complete_Sample'
    documents['pathSave'] = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work'
    documents['cluster_prefix'] = 'CE_'
    documents['totalClusters'] = 30
    documents['clusterIDAllowed'] = np.linspace(0, documents['totalClusters'] - 1, documents['totalClusters'], dtype=np.int)
    documents['subjectsAllowed'] = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
    documents['zcat'] = {
                'z_value'   :
                    ['z014p003', 'z006p772', 'z004p614', 'z003p512', 'z002p825',
                     'z002p348', 'z001p993', 'z001p716', 'z001p493', 'z001p308',
                     'z001p151', 'z001p017', 'z000p899', 'z000p795', 'z000p703',
                     'z000p619', 'z000p543', 'z000p474', 'z000p411', 'z000p366',
                     'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155',
                     'z000p113', 'z000p101', 'z000p073', 'z000p036', 'z000p000'][::-1],
                'z_IDNumber':
                    ['000', '001', '002', '003', '004', '005', '006', '007', '008',
                     '009', '010', '011', '012', '013', '014', '015', '016', '017',
                     '018', '019', '020', '021', '022', '023', '024', '025', '026',
                     '027', '028', '029'][::-1]}
    documents['redshiftAllowed'] = documents['zcat']['z_value']
    documents['centralFOF_groupNumber'] = 1
    documents['sample_completeness'] = f"{documents['simulation_name']}_sample_completeness.npy"
    with open(f"{documents['simulation_name']}.yaml", 'w') as file:
        yaml.dump(eval(documents), file)

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
