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

def global_setup_yaml():
    thisfile_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(thisfile_path, 'glob.yaml'), 'w') as file:
        dict_file = {}
        dict_file['setup'] = 'volume'
        documents = yaml.dump(dict_file, file)
        print(f'[+] Creating global info file: glob.yaml. Contents:')
        for item, doc in documents.items():
            print("[+]\t", item, ":", doc)


# def bahamas_mass_cut():
    #     cluster = Cluster(simulation_name='bahamas',
    #                     clusterID=0,
    #                     redshift='z000p000',
    #                     comovingframe=False,
    #                     fastbrowsing=True)
    #     n_largeM = 0
    #     n_total = 0
    #     gn_tot = np.zeros(0, dtype=np.int)
    #     N_halos = 0
    #     for counter, file in enumerate(cluster.groups_filePaths()):
    #         print(f"[+] Analysing eagle_subfind_tab file {counter}")
    #         with h5py.File(file, 'r') as group_file:
    #             m500 = group_file['/FOF/Group_M_Crit500'][:] * 10 ** 10
    #             n_total += len(m500)
    #             m_filter = np.where(m500 > 10 ** 13)[0] + N_halos
    #             gn_tot = np.append(gn_tot, m_filter)
    #             n_largeM += len(m_filter)
    #             N_halos += group_file['Header'].attrs['Ngroups']
    #
    #     print('n_largeM:', n_largeM)
    #     print('n_total:', n_total)
    #     print('N_halos:', N_halos)
    #     print('gn_tot:', gn_tot)
    #     np.save(f'import_toolkit/bahamas_fofnumber_list_10--13.npy', gn_tot)
    #
    # bahamas_mass_cut()


    # def check_dirs(self) -> np.ndarray:
    #     """
    #     Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    #     are present in the simulation archive.
    #     :return:
    #     """
    #     iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
    #     check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
    #     for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
    #         c = Cluster(simulation_name=self.simulation_name,
    #                     clusterID=halo_id,
    #                     redshift=halo_z)
    #
    #         redshift_threshold = redshift_str2num(halo_z) < 1.8
    #         test = c.is_cluster() * c.is_redshift() * redshift_threshold
    #         check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test
    #         print(process_n, halo_id, halo_z, test)
    #
    #     np.save(f'import_toolkit/{c.simulation_name}_sample_completeness.npy', check_matrix)
    #     print(len(np.where(check_matrix == True)[0])/np.product(check_matrix.shape)*100)
    #     for i in check_matrix:
    #         print([int(j) for j in i])
    #     return check_matrix
    #
    # s = Simulation(simulation_name='macsis')
    # for n in s.clusterIDAllowed:
    #     path1 = os.path.join(s.pathSave, s.simulation_name,
    #                         f'halo{s.halo_Num(n)}', f'halo{s.halo_Num(n)}_z000p024')
    #     path2 = os.path.join(s.pathSave, s.simulation_name,
    #                          f'halo{s.halo_Num(n)}', f'halo{s.halo_Num(n)}_z000p240')
    #     os.rename(path1, path2)