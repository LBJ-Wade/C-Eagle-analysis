from __future__ import print_function, division, absolute_import

import numpy as np
import h5py as h5
import os
from functools import wraps

# Local modules
import redshift_catalogue_ceagle as zcat
import _cluster_retriever
import _cluster_profiler

# Import global methods separately
from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str


#################################
#                               #
#	   S I M U L A T I O N      #
# 			C L A S S           #
#							    #
#################################

class Simulation():

    def __init__(self, simulation_name = 'celr_b'):

        self.pathSave = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work'
        self.particle_type_conversion = {        'gas': '0',
                                         'dark_matter': '1',
                                               'stars': '4',
                                         'black_holes': '5'}

        if simulation_name == 'ceagle':
            self.simulation = 'C-EAGLE'
            self.computer = 'cosma.dur.ac.uk'
            self.pathData = '/cosma5/data/dp004/C-EAGLE/Complete_Sample'
            self.cluster_prefix = 'CE_'
            self.totalClusters = 30
            self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
            self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
            self.redshiftAllowed = zcat.group_data()['z_value']
            self.centralFOF_groupNumber = 1

        elif simulation_name == 'celr_b':
            self.simulation = 'CELR-bahamas'
            self.computer = 'cosma.dur.ac.uk'
            self.pathData = '/cosma5/data/dp004/dc-pear3/data/bahamas'
            self.cluster_prefix = 'halo_'
            self.totalClusters = 45
            self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
            self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
            self.zcat = {
                'z_value':
                    ['z001p017', 'z000p899', 'z000p795', 'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411',
                     'z000p366', 'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101',
                     'z000p073', 'z000p036', 'z000p000'],
                'z_IDNumber':
                    ['011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
                     '025', '026', '027', '028', '029']}
            self.redshiftAllowed = self.zcat['z_value']
            self.centralFOF_groupNumber = 1

        elif simulation_name == 'celr_e':
            self.simulation = 'CELR-eagle'
            self.computer = 'cosma.dur.ac.uk'
            self.pathData = '/cosma5/data/dp004/dc-pear3/data/eagle'
            self.cluster_prefix = 'halo_'
            self.totalClusters = 45
            self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
            self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
            self.zcat = {
                'z_value':
                    ['z001p017', 'z000p899', 'z000p795', 'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411',
                     'z000p366', 'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101',
                     'z000p073', 'z000p036', 'z000p000'],
                'z_IDNumber':
                    ['011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
                     '025', '026', '027', '028', '029']}
            self.redshiftAllowed = self.zcat['z_value']
            self.centralFOF_groupNumber = 1

        elif simulation_name == 'macsis':
            self.simulation = 'MACSIS'
            self.computer = 'virgo_nas@mizar.jb.man.ac.ukk'
            self.pathData = ''
            self.cluster_prefix = 'halo_'
            self.totalClusters = 45
            self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
            self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
            self.zcat = {
                'z_value':
                    ['z001p017', 'z000p899', 'z000p795', 'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411',
                     'z000p366', 'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101',
                     'z000p073', 'z000p036', 'z000p000'],
                'z_IDNumber':
                    ['011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
                     '025', '026', '027', '028', '029']}
            self.redshiftAllowed = self.zcat['z_value']
            self.centralFOF_groupNumber = 1

        elif simulation_name == 'bahamas':
            self.simulation = 'BAHAMAS'
            self.computer = 'virgo_nas@mizar.jb.man.ac.uk'
            self.pathData = ''
            self.cluster_prefix = 'halo_'
            self.totalClusters = 45
            self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
            self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
            self.zcat = {
                'z_value':
                    ['z001p017', 'z000p899', 'z000p795', 'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411',
                     'z000p366', 'z000p352', 'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101',
                     'z000p073', 'z000p036', 'z000p000'],
                'z_IDNumber':
                    ['011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
                     '025', '026', '027', '028', '029']}
            self.redshiftAllowed = self.zcat['z_value']
            self.centralFOF_groupNumber = 1

        else:
            raise(ValueError("Simulation name error: expected [`ceagle` or, `celr_b` or, `celr_e` or, `macsis` "
                             "or, `bahamas`], got {}.".format(simulation_name)))



    def set_pathData(self, newPath: str):
        self.pathData = newPath

    def set_totalClusters(self, newNumber: int):
        self.totalClusters = newNumber

    def get_redshiftAllowed(self, dtype=float):
        """	Access the allowed redshifts in the simulation.	"""
        if dtype == str:
            return self.redshiftAllowed
        if dtype == float:
            return [redshift_str2num(z) for z in self.redshiftAllowed]

    def info(self):
        for attr in dir(self):
            print("obj.%s = %r" % (attr, getattr(self, attr)))



#################################
#                               #
#		  C L U S T E R  	    #
# 			C L A S S           #
#							    #
#################################

class Cluster(Simulation,
              _cluster_retriever.Mixin,
              _cluster_profiler.Mixin):

    def __init__(self, *args, clusterID: int = 0, redshift: str = 0.0, **kwargs):
        super().__init__()

        # Initialise and validate attributes
        self.set_clusterID(clusterID)
        self.redshift = redshift

        # Pass attributed into kwargs
        kwargs['clusterID'] = self.clusterID
        kwargs['redshift'] = self.redshift

        # Set additional attributes from methods
        # self.hubble_param = self.file_hubble_param()
        # self.comic_time = self.file_comic_time()
        # self.redshift = self.file_redshift()
        # self.OmegaBaryon = self.file_OmegaBaryon()
        # self.Omega0 = self.file_Omega0()
        # self.OmegaLambda = self.file_OmegaLambda()

    # Change and validate Cluster attributes
    def set_clusterID(self, clusterID: int):
        try:
            assert (type(clusterID) is int), 'clusterID must be integer.'
            assert (clusterID in self.clusterIDAllowed), 'clusterID out of bounds (00 ... 29).'
        except AssertionError:
            raise
        else:
            self.clusterID = clusterID

    def path_from_cluster_name(self):
        """
        RETURNS: string type. Path of the hdf5 file to extract data from.
        """
        # os.chdir(sys.path[0])	# Set working directory as the directory of this file.
        master_directory = self.pathData
        cluster_ID = self.cluster_prefix + halo_Num(self.clusterID)
        data_dir = 'data'
        return os.path.join(master_directory, cluster_ID, data_dir)

    def file_hubble_param(self):
        """
        AIM: retrieves the Hubble parameter of the file
        RETURNS: type = double
        """
        _, attr_value = self.extract_header_attribute_name('HubbleParam')
        return attr_value

    def file_comic_time(self):
        _, attr_value = self.extract_header_attribute_name('Time')
        return attr_value

    def file_redshift(self):
        _, attr_value = self.extract_header_attribute_name('Redshift')
        return attr_value

    def file_OmegaBaryon(self):
        _, attr_value = self.extract_header_attribute_name('OmegaBaryon')
        return attr_value

    def file_Omega0(self):
        _, attr_value = self.extract_header_attribute_name('Omega0')
        return attr_value

    def file_OmegaLambda(self):
        _, attr_value = self.extract_header_attribute_name('OmegaLambda')
        return attr_value

    def file_Ngroups(self):
        _, attr_value = self.extract_header_attribute_name('TotNgroups')
        return attr_value

    def file_Nsubgroups(self):
        _, attr_value = self.extract_header_attribute_name('TotNsubgroups')
        return attr_value

    def file_MassTable(self):
        _, attr_value = self.extract_header_attribute_name('MassTable')
        return attr_value

    def file_NumPart_Total(self):
        """
        [
            NumPart_Total(part_type0),
            NumPart_Total(part_type1),
            NumPart_Total(part_type2),
            NumPart_Total(part_type3),
            NumPart_Total(part_type4),
            NumPart_Total(part_type5)
        ]

        :return: array of 6 elements
        """
        _, attr_value = self.extract_header_attribute_name('NumPart_Total')
        return attr_value

    def DM_particleMass(self):
        return self.file_MassTable()[1]

    def DM_NumPart_Total(self):
        return self.file_NumPart_Total()[1]


    @staticmethod
    def type_convert(part_type):
        """
        AIM: returns a string characteristic of the particle type selected
        RETURNS: string of number 0<= n <= 5
        """
        if part_type == 'gas' or part_type == 0 or part_type == '0':
            return '0'
        elif part_type == 'highres_DM' or part_type == 1 or part_type == '1':
            return '1'
        elif part_type == 'lowres_DM' or part_type == 2 or part_type == '2':
            return '2'
        elif part_type == 'lowres_DM' or part_type == 3 or part_type == '3':
            return '3'
        elif part_type == 'stars' or part_type == 4 or part_type == '4':
            return '4'
        elif part_type == 'black_holes' or part_type == 5 or part_type == '5':
            return '5'
        else:
            print("[ERROR] You entered the wrong particle type!")
            exit(1)




