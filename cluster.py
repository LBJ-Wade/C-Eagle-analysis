from __future__ import print_function, division, absolute_import

import numpy as np
import h5py as h5
import os
from functools import wraps

# Local modules
import redshift_catalogue_ceagle as zcat
import _cluster_retriever
import _cluster_profiler


#################################
#                               #
# 	G L O B    M E T H O D S    #
#							    #
#################################

def halo_Num(n: int):
    """
    Returns the halo number in format e.g. 00, 01, 02
    """
    return '%02d' % (n,)


def redshift_str2num(z: str):
    """
    Converts the redshift of the snapshot from text to numerical,
    in a format compatible with the file names.
    E.g. float z = 2.16 <--- str z = 'z002p160'.
    """
    z = z.strip('z').replace('p', '.')
    return round(float(z), 3)


def redshift_num2str(z: float):
    """
    Converts the redshift of the snapshot from numerical to
    text, in a format compatible with the file names.
    E.g. float z = 2.16 ---> str z = 'z002p160'.
    """
    integer_z, decimal_z = str(z).split('.')
    return 'z' + integer_z.ljust(3, '0') + 'p' + decimal_z.rjust(3, '0')


def free_memory(var_list, invert=False):
    """
    Function for freeing memory dynamically.
    invert allows to delete all local variables that are NOT in var_list.
    """
    if not invert:
        for name in var_list:
            if not name.startswith('_') and name in dir():
                del globals()[name]
    if invert:
        for name in dir():
            if name in var_list and not name.startswith('_'):
                del globals()[name]


#################################
#                               #
#	   S I M U L A T I O N      #
# 			C L A S S           #
#							    #
#################################

class Simulation:

    def __init__(self):
        self.simulation = 'C-EAGLE'
        self.computer = 'cosma.dur.ac.uk'
        self.pathData = '/cosma5/data/dp004/C-EAGLE/Complete_Sample'
        self.pathSave = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work'
        self.totalClusters = 30
        self.clusterIDAllowed = np.linspace(0, self.totalClusters - 1, self.totalClusters, dtype=np.int)
        self.subjectsAllowed = ['particledata', 'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
        self.redshiftAllowed = zcat.group_data()['z_value']

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


#################################
#                               #
#		  C L U S T E R  	    #
# 			C L A S S           #
#							    #
#################################

class Cluster(Simulation,
              _cluster_retriever.Mixin,
              _cluster_profiler.Mixin):

    def __init__(self, *args, clusterID: int = 0, redshift: float = 0.0, **kwargs):
        super().__init__()

        # Initialise and validate attributes
        self.set_clusterID(clusterID)
        self.set_redshift(redshift)

        # Pass attributed into kwargs
        kwargs['clusterID'] = self.clusterID
        kwargs['redshift'] = self.redshift

        # Set additional attributes from methods
        self.hubble_param = self.file_hubble_param()
        self.comic_time = self.file_comic_time()
        self.redshift = self.file_redshift()
        # self.Ngroups = self.file_Ngroups()
        # self.Nsubgroups = self.file_Nsubgroups()
        self.OmegaBaryon = self.file_OmegaBaryon()
        self.Omega0 = self.file_Omega0()
        self.OmegaLambda = self.file_OmegaLambda()

    # Change and validate Cluster attributes
    def set_clusterID(self, clusterID: int):
        try:
            assert (type(clusterID) is int), 'clusterID must be integer.'
            assert (clusterID in self.clusterIDAllowed), 'clusterID out of bounds (00 ... 29).'
        except AssertionError:
            raise
        else:
            self.clusterID = clusterID

    def set_redshift(self, redshift: float):
        try:
            assert (type(redshift) is float), 'redshift must be float.'
            assert (redshift >= 0.0), 'Negative redshift.'
            assert (redshift_num2str(redshift) in self.redshiftAllowed), 'Redshift not valid.'
        except AssertionError:
            raise
        else:
            self.redshift = round(redshift, 3)

    def path_from_cluster_name(self):
        """
        RETURNS: string type. Path of the hdf5 file to extract data from.
        """
        # os.chdir(sys.path[0])	# Set working directory as the directory of this file.
        master_directory = self.pathData
        cluster_ID = 'CE_' + halo_Num(self.clusterID)
        data_dir = 'data'
        return os.path.join(master_directory, cluster_ID, data_dir)

    #####################################################
    #													#
    #				D E C O R A T O R S  				#
    # 									 				#
    #####################################################

    def data_subject(**decorator_kwargs):
        """
        This decorator adds functionality to the data import functions.
        It dynamically creates attributes and filepaths, to be then
        passed onto the actual import methods.
        :param decorator_kwargs: subject = (str)
        :return: decorated function with predefined **kwargs
        The **Kwargs are dynamically allocated to the external methods.
        """

        # print("Reading ", decorator_kwargs['subject'], " files.")

        def wrapper(f):  # a wrapper for the function
            @wraps(f)
            def decorated_function(self, *args, **kwargs):  # the decorated function

                redshift_str = redshift_num2str(round(self.redshift, 3))
                redshift_i = self.redshiftAllowed.index(redshift_str)
                redshift_index = zcat.group_data()['z_IDNumber'][redshift_i]
                sbj_string = decorator_kwargs['subject'] + '_' + redshift_index + '_' + redshift_str
                file_dir = os.path.join(self.path_from_cluster_name(), sbj_string)
                file_list = os.listdir(file_dir)

                if decorator_kwargs['subject'] == 'particledata':
                    prefix = 'eagle_subfind_particles_'
                elif decorator_kwargs['subject'] == 'groups':
                    prefix = 'eagle_subfind_tab_'
                elif decorator_kwargs['subject'] == 'snapshot':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'snipshot':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'hsmldir':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'groups_snip':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")

                # Transfer function state into the **kwargs
                # These **kwargs are accessible to the decorated class methods
                kwargs['subject'] = decorator_kwargs['subject']
                kwargs['file_dir'] = file_dir
                kwargs['file_list'] = [x for x in file_list if x.startswith(prefix)]
                kwargs['file_list_sorted'] = sorted([os.path.join(file_dir, file) for file in kwargs['file_list']])

                return f(self, *args, **kwargs)

            return decorated_function

        return wrapper

    @data_subject(subject="particledata")
    def extract_header_attribute(self, element_number, *args, **kwargs):
        # Import data from hdf5 file
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/Header"]
        attr_name = list(h5dset.attrs.keys())[element_number]
        attr_value = list(h5dset.attrs.values())[element_number]
        h5file.close()
        return attr_name, attr_value

    @data_subject(subject="particledata")
    def extract_header_attribute_name(self, element_name, *args, **kwargs):
        # Import data from hdf5 file
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/Header"]
        attr_name = h5dset.attrs.get(element_name, default=None)
        attr_value = h5dset.attrs.get(element_name, default=None)
        h5file.close()
        return attr_name, attr_value

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




