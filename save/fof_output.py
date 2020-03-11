"""
------------------------------------------------------------------
FILE:   fof_output.py
AUTHOR: Edo Altamura
DATE:   25-11-2019
------------------------------------------------------------------
In order to make the data post-processing more manageable, instead
of calculating quantities from the simulations every time, just
compute them once and store them in a hdf5 file.
This process of data reduction level condenses data down to a few KB
or MB and it is possible to transfer it locally for further analysis.
-------------------------------------------------------------------
"""

from mpi4py import MPI
import itertools
import numpy as np
import sys
import os.path
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from _cluster_retriever import redshift_str2num, redshift_num2str
from testing import angular_momentum
from testing import mergers
import progressbar
import save

__HDF5_SUBFOLDER__ = 'FOF'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#####################################################
#													#
#				D E C O R A T O R S  				#
# 									 				#
#####################################################

def make_parallel_MPI(function):
    """
    This decorator adds functionality to the processing routing for the whole
    simulation. It creates a list of processes to initialise, each taking a
    halo at a redshifts. It then allocates dynamically each process to a idle
    CPU in a recursive way using the modulus function. Ae each iteration it
    executes the function to be wrapped with *args and **kwargs.

    :param decorator_kwargs: simulation_name = (str)
    :param decorator_kwargs: out_allPartTypes = (bool)
    :return: decorated function with predefined **kwargs

    The **Kwargs are dynamically allocated to the external methods.
    """

    def wrapper(*args, **kwargs):

        # Checks that the essential paremeters are there
        assert not kwargs['out_allPartTypes'] is None
        assert not kwargs['simulation_name'] is None

        # Generate a simulation object and oush it to **kwargs
        sim = Simulation(simulation_name=kwargs['simulation_name'])
        kwargs['simulation'] = sim

        # Set-up the MPI allocation schedule
        process = 0
        process_iterator = itertools.product(sim.clusterIDAllowed, sim.redshiftAllowed)

        for halo_num, redshift in process_iterator:

            if process % size == rank:

                cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
                file_name = sim.cluster_prefix + sim.halo_Num(halo_num) + redshift
                fileCompletePath = sim.pathSave + '/' + sim.simulation + '_output/collective_output/' + file_name + '.hdf5'

                kwargs['cluster'] = cluster_obj
                kwargs['fileCompletePath'] = fileCompletePath

                print('CPU ({}/{}) is processing halo {} @ z = {} ------ process ID: {}'.format(rank, size, cluster_obj.clusterID, cluster_obj.redshift, process))
                # Each CPU loops over all apertures - this avoids concurrence in file reading
                # The loop over apertures is defined explicitly in the wrapped function.
                function(*args, **kwargs)

            process += 1

    return wrapper

class FOFOutput(save.SimulationOutput):

    def __init__(self,
                 cluster: Cluster,
                 filename: str = None,
                 data: dict = None,
                 attrs: dict = None):
        """
        Initialization of the FOFOutput class.

        :param cluster: expect Cluster object
            The FOFOutput is unique to the Cluster object and communicates with the output directory with
            specific halo number and redshift.

        :param filename: expect str
            The filename should end with `.hdf5` and represents the output file for storing the computed
            data.

        :param data: expect dict
            The data keyword accepts a dictionary for the input data> It should be of the form:
            {'internal/path/groups/dataset_name' : data_numpy_array,
             'internal/path/groups/dataset_name' : data_numpy_array,
                            ...                  :     ...         }
            The input of a dictionary is optional (default set to None).

        :param attrs: expect dict
            The attributes parsed into the attrs keyword are associated to the whole file. The dictionary
            should be of the format:
            {`Description` : ```Something that describe the datasets```,
             `Units`       : `meters second^-1`}
        """

        super(save.SimulationOutput, self).__init__(simulation_name=cluster.simulation_name)

        assert filename.endswith('.hdf5'), f"Filename must end with `.hdf5` extension. Got {filename} instead."

        self.filename = filename
        self.data = data
        self.attrs = attrs
        self.FOFDirectory = os.path.join(cluster.pathSave,
                                        cluster.simulation_name,
                                        f'halo{self.halo_Num(cluster.clusterID)}',
                                        f'halo{self.halo_Num(cluster.clusterID)}_{cluster.redshift}')

    def get_filename(self):
        return self.filename

    def get_directory(self):
        return self.FOFDirectory

    def get_data(self):
        return self.data

    def get_attrs(self):
        return self.attrs

    @staticmethod
    def groups_from_path(internal_path: str):
        l = internal_path.split('/')
        return list(filter(lambda a: a is not '', l))

    @staticmethod
    def split_data_dict(data_dict: dict):
        return data_dict.keys(), data_dict.values()

    def make_hdf5file(self, dataset_paths: list = None, dataset_values: list = None) -> None:
        """
        Function that makes a new hdf5 file and pushes data into it. It is to be used in conjunction with
        the split_data_dict static method and the self.data attribute member.

        :param dataset_paths: expect list of strings with `/` separators
            This is the list of paths internal to the hdf5 file from the base directory to the dataset
            name. Example:
            ['internal/path/groups/dataset_name',
             'internal/path/groups/dataset_name',
                            ...                 ]
            If only one path needed, parse ['only/internal/path/groups/dataset_name'].
            The list must contain at least one path or one dataset and tha string cannot end with a separator,
            as this would indicate an internal directory with no dataset name specified.

        :param dataset_values: expect list of np.arrays
            This is the list of np.arrays, uniquely matched to the datasets internal paths.
            The format is similar to the dataset path argument. Since they should come from `self.data` attribute,
            they should have the same length by definition.

        :return: None
        """

        assert dataset_paths is not None and len(dataset_paths) != 0, ("`dataset_path` not valid. Expected at least " \
                                                                      f"one list element, got {len(dataset_paths)}.")

        assert dataset_values is not None and len(dataset_values) != 0, ("`dataset_values` not valid. Expected at least " \
                                                                           f"one list element, got {len(dataset_values)}.")


        # Remove file if already exists and create a new one
        if os.path.isfile(os.path.join(self.FOFDirectory, self.filename)):
            os.remove(os.path.join(self.FOFDirectory, self.filename))
            print(f'[ FOFOutput ]\t==> Removed old {self.filename} file.')

        # Create file and optional groups within it
        FOFfile = h5py.File(os.path.join(self.FOFDirectory, self.filename), 'w')
        print(f'[ FOFOutput ]\t==> Created new {self.filename} file.')

        # Push the attributes to file, if any
        if self.attrs is not None and len(self.attrs.keys()) > 0:
            for key, text in zip(self.attrs.keys(), self.attrs.values()):
                FOFfile.attrs[key] = text

        for internal_path, dataset_content in zip(dataset_paths, dataset_values):

            assert not internal_path.endswith('/'), "Invalid hdf5 internal path"
            assert type(dataset_content) is np.ndarray, "Can only push numpy.ndarrays into hdf5 files."

            nested_groups = self.groups_from_path(internal_path)
            if len(nested_groups) == 1:
                FOFfile.create_dataset(nested_groups[0], data=dataset_content)
            else:
                for nested_group in nested_groups[:-1]:
                    g = FOFfile.create_group(nested_group)
                    g.create_dataset(nested_groups[-1], data = dataset_content)

            print(f'[ FOFOutput ]\t==> Created {internal_path} dataset in {self.filename} file.')

        FOFfile.close()


    def add_to_hdf5file(self, dataset_paths: list = None, dataset_values: list = None) -> None:
        """
        Function that takes an existing hdf5 file and appends new datasets into it. It is to be used in conjunction with
        the split_data_dict static method and the self.data attribute member.

        :param dataset_paths: expect list of strings with `/` separators
            This is the list of paths internal to the hdf5 file from the base directory to the dataset
            name. Example:
            ['internal/path/groups/dataset_name',
             'internal/path/groups/dataset_name',
                            ...                 ]
            If only one path needed, parse ['only/internal/path/groups/dataset_name'].
            The list must contain at least one path or one dataset and tha string cannot end with a separator,
            as this would indicate an internal directory with no dataset name specified.

        :param dataset_values: expect list of np.arrays
            This is the list of np.arrays, uniquely matched to the datasets internal paths.
            The format is similar to the dataset path argument. Since they should come from `self.data` attribute,
            they should have the same length by definition.

        :return: None
        """

        assert dataset_paths is not None and len(dataset_paths) != 0, ("`dataset_path` not valid. Expected at least " \
                                                                        f"one list element, got {len(dataset_paths)}.")

        assert dataset_values is not None and len(dataset_values) != 0, ("`dataset_values` not valid. Expected at least " \
                                                                        f"one list element, got {len(dataset_values)}.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, self.filename)), f"Target hdf5 file must exist in {self.FOFDirectory}"

        # Open file and optional groups within it
        FOFfile = h5py.File(os.path.join(self.FOFDirectory, self.filename), 'r+')
        print(f'[ FOFOutput ]\t==> Opening {self.filename} file.')

        for internal_path, dataset_content in zip(dataset_paths, dataset_values):

            assert not internal_path.endswith('/'), "Invalid hdf5 internal path"
            assert type(dataset_content) is np.ndarray, "Can only push numpy.ndarrays into hdf5 files."

            nested_groups = self.groups_from_path(internal_path)
            if len(nested_groups) == 1:
                FOFfile.create_dataset(nested_groups[0], data=dataset_content)
            else:
                for nested_group in nested_groups[:-1]:
                    g = FOFfile.create_group(nested_group)
                    g.create_dataset(nested_groups[-1], data=dataset_content)

            print(f'[ FOFOutput ]\t==> Created {internal_path} dataset in {self.filename} file.')

        FOFfile.close()


    def makefile(self):
        data = self.split_data_dict(self.data)
        self.make_hdf5file(dataset_paths = list(data[0]),  dataset_values = list(data[1]))

    def push(self, additional_datasets):
        data = self.split_data_dict(additional_datasets)
        self.add_to_hdf5file(dataset_paths = list(data[0]),  dataset_values = list(data[1]))





class FOFDatagen(save.SimulationOutput):

    def __init__(self, cluster: Cluster):
        super(save.SimulationOutput, self).__init__(simulation_name=cluster.simulation_name)
        self.cluster = cluster
        self.FOFDirectory = os.path.join(cluster.pathSave,
                                         cluster.simulation_name,
                                         f'halo{self.halo_Num(cluster.clusterID)}',
                                         f'halo{self.halo_Num(cluster.clusterID)}_{cluster.redshift}')

    def push_R_crit(self):
        data = {'/R_200_crit' : np.array([self.cluster.r200]),
                '/R_500_crit' : np.array([self.cluster.r500]),
                '/R_2500_crit' : np.array([self.cluster.r2500])}
        attributes = {'Description' : 'R_crits',
                      'Units' : 'Mpc'}
        out = FOFOutput(self.cluster, filename = 'rcrit.hdf5', data = data, attrs = attributes)
        out.makefile()

    def push_apertures(self):

        data = {'/Apertures': np.array(self.cluster.generate_apertures())}

        attributes = {'Description': 'Aperture radii in comoving coordinates',
                      'Units': 'Mpc'}

        out = FOFOutput(self.cluster, filename='apertures.hdf5', data=data, attrs=attributes)
        out.makefile()

    def push_mass(self):

        part_mass = np.zeros((0,4), dtype=np.float)
        total_mass = np.zeros((0,), dtype=np.float)

        for r in self.cluster.generate_apertures():

            part_mass_aperture = self.cluster.group_mass_aperture(out_allPartTypes=True, aperture_radius=r)
            part_mass = np.concatenate((part_mass, [part_mass_aperture]), axis=0)

            tot_mass_aperture = np.sum(part_mass_aperture)
            total_mass = np.concatenate((total_mass, [tot_mass_aperture]), axis=0)

        data = {'/Total_mass': np.array(total_mass),
                '/ParType0_mass' : np.array(part_mass)[:,0],
                '/ParType1_mass' : np.array(part_mass)[:,1],
                '/ParType4_mass' : np.array(part_mass)[:,2],
                '/ParType5_mass' : np.array(part_mass)[:,3]}

        attributes = {'Description': """The ParType_mass array contains the mass enclosed within a given aperture, 
        for each particle type (in the order 0, 1, 4, 5).
        The Total-mass array gives the total mass within an aperture of all partTypes summed together.""",
                      'Units': '10^10 M_sun'}

        out = FOFOutput(self.cluster, filename='mass.hdf5', data=data, attrs=attributes)
        out.makefile()

    def push_centre_of_mass(self):

        Total_CoM    = np.zeros((0,3), dtype=np.float)
        ParType0_CoM = np.zeros((0,3), dtype=np.float)
        ParType1_CoM = np.zeros((0,3), dtype=np.float)
        ParType4_CoM = np.zeros((0,3), dtype=np.float)
        ParType5_CoM = np.zeros((0,3), dtype=np.float)

        for r in self.cluster.generate_apertures():

            part_CoM_aperture, _mass = self.cluster.group_centre_of_mass(aperture_radius=r, out_allPartTypes=True)
            ParType0_CoM = np.concatenate((ParType0_CoM, [part_CoM_aperture[0]]), axis = 0)
            ParType1_CoM = np.concatenate((ParType1_CoM, [part_CoM_aperture[1]]), axis = 0)
            ParType4_CoM = np.concatenate((ParType4_CoM, [part_CoM_aperture[2]]), axis = 0)
            ParType5_CoM = np.concatenate((ParType5_CoM, [part_CoM_aperture[3]]), axis = 0)

            Total_CoM_aperture, _totmass = self.cluster.centre_of_mass(_mass, part_CoM_aperture)
            Total_CoM = np.concatenate((Total_CoM, [Total_CoM_aperture]), axis=0)

        data = {'/Total_CoM'    : np.array(Total_CoM),
                '/ParType0_CoM' : np.array(ParType0_CoM),
                '/ParType1_CoM' : np.array(ParType1_CoM),
                '/ParType4_CoM' : np.array(ParType4_CoM),
                '/ParType5_CoM' : np.array(ParType5_CoM)}

        attributes = {'Description': """Datasets with the (x,y,z) cartesian coordinates of the Centre of Mass of the 
        cluster, computed based on particles within a specific aperture radius from the Centre of Potential.
        The datasets contain the CoM for each high-res particle type and also the combined total value of all particles.""",
                      'Units': '[Mpc], [Mpc], [Mpc]'}

        out = FOFOutput(self.cluster, filename='centre_of_mass.hdf5', data=data, attrs=attributes)
        out.makefile()

    def push_peculiar_velocity(self):

        Total_ZMF    = np.zeros((0,3), dtype=np.float)
        ParType0_ZMF = np.zeros((0,3), dtype=np.float)
        ParType1_ZMF = np.zeros((0,3), dtype=np.float)
        ParType4_ZMF = np.zeros((0,3), dtype=np.float)
        ParType5_ZMF = np.zeros((0,3), dtype=np.float)

        for r in self.cluster.generate_apertures():

            part_ZMF_aperture, _mass = self.cluster.group_zero_momentum_frame(aperture_radius=r, out_allPartTypes=True)
            ParType0_ZMF = np.concatenate((ParType0_ZMF, [part_ZMF_aperture[0]]), axis = 0)
            ParType1_ZMF = np.concatenate((ParType1_ZMF, [part_ZMF_aperture[1]]), axis = 0)
            ParType4_ZMF = np.concatenate((ParType4_ZMF, [part_ZMF_aperture[2]]), axis = 0)
            ParType5_ZMF = np.concatenate((ParType5_ZMF, [part_ZMF_aperture[3]]), axis = 0)

            Total_ZMF_aperture, _totmass = self.cluster.zero_momentum_frame(_mass, part_ZMF_aperture)
            Total_ZMF = np.concatenate((Total_ZMF, [Total_ZMF_aperture]), axis=0)

        data = {'/Total_ZMF'    : np.array(Total_ZMF),
                '/ParType0_ZMF' : np.array(ParType0_ZMF),
                '/ParType1_ZMF' : np.array(ParType1_ZMF),
                '/ParType4_ZMF' : np.array(ParType4_ZMF),
                '/ParType5_ZMF' : np.array(ParType5_ZMF)}

        attributes = {'Description': """Datasets with the peculiar linear bulk velocity of the cluster, calculated 
        from particles within a specific aperture radius from the Centre of Potential. Individual datasets contain peculiar velocity information 
        about each particle type separately, as well as one with combined total contribution.""",
                      'Units': '[km s^-1], [km s^-1], [km s^-1]'}

        out = FOFOutput(self.cluster, filename='peculiar_velocity.hdf5', data=data, attrs=attributes)
        out.makefile()


    def push_angular_momentum(self):

        Total_angmom    = np.zeros((0, 3), dtype=np.float)
        ParType0_angmom = np.zeros((0, 3), dtype=np.float)
        ParType1_angmom = np.zeros((0, 3), dtype=np.float)
        ParType4_angmom = np.zeros((0, 3), dtype=np.float)
        ParType5_angmom = np.zeros((0, 3), dtype=np.float)

        for r in self.cluster.generate_apertures():
            part_angmom_aperture, _mass = self.cluster.group_angular_momentum(aperture_radius=r, out_allPartTypes=True)
            ParType0_angmom = np.concatenate((ParType0_angmom, [part_angmom_aperture[0]]), axis=0)
            ParType1_angmom = np.concatenate((ParType1_angmom, [part_angmom_aperture[1]]), axis=0)
            ParType4_angmom = np.concatenate((ParType4_angmom, [part_angmom_aperture[2]]), axis=0)
            ParType5_angmom = np.concatenate((ParType5_angmom, [part_angmom_aperture[3]]), axis=0)

            Total_angmom_aperture = np.sum(part_angmom_aperture, axis=0)
            Total_angmom = np.concatenate((Total_angmom, [Total_angmom_aperture]), axis=0)

        data = {'/Total_angmom'   : np.array(Total_angmom),
                '/ParType0_angmom': np.array(ParType0_angmom),
                '/ParType1_angmom': np.array(ParType1_angmom),
                '/ParType4_angmom': np.array(ParType4_angmom),
                '/ParType5_angmom': np.array(ParType5_angmom)}

        attributes = {'Description': """Datasets with the angular momentum vector of the cluster, calculated 
        from particles within a specific aperture radius from the Centre of Potential. Individual datasets contain 
        angular momentum information about each particle type separately, as well as one with combined total contribution.""",
                      'Units': '[10^10 M_sun * km * s^-1 * Mpc], [10^10 M_sun * km * s^-1 * Mpc], [10^10 M_sun * km * s^-1 * Mpc]'}

        out = FOFOutput(self.cluster, filename='angular_momentum.hdf5', data=data, attrs=attributes)
        out.makefile()


    def push_dynamical_merging_index(self):
        """
        Computes the dynamical index based on the centre of potential coordinates, the centre of mass coordinates
        and the aperture radius.

        dynamical_merging_index = || CoM(r) - CoP(r) || / r

        The calculation of the dynamical_merging_index assumes the CoM and aperture data are already generated as
        partial results.

        :return: None
        """
        assert os.path.isfile(os.path.join(self.FOFDirectory, 'centre_of_mass.hdf5')), ("Centre of mass data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have "
                                                                                        "already been computed for "
                                                                                        "this cluster and this "
                                                                                        "redshift")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                   f"found in {self.FOFDirectory}."
                                                                                   "Check that they have already been "
                                                                                   "computed for this cluster and this redshift.")

        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        # Read centre_of_mass data
        with h5py.File(os.path.join(self.FOFDirectory, 'centre_of_mass.hdf5'), 'r') as input_file:
            Total_CoM    = np.array(input_file.get('Total_CoM'))
            ParType0_CoM = np.array(input_file.get('ParType0_CoM'))
            ParType1_CoM = np.array(input_file.get('ParType1_CoM'))
            ParType4_CoM = np.array(input_file.get('ParType4_CoM'))
            ParType5_CoM = np.array(input_file.get('ParType5_CoM'))

        CoP = np.ones((len(apertures), 3)) * self.cluster.group_centre_of_potential().reshape((1,3))

        Total_dynindex    = self.cluster.dynamical_merging_index(CoP, Total_CoM, apertures)
        ParType0_dynindex = self.cluster.dynamical_merging_index(CoP, ParType0_CoM, apertures)
        ParType1_dynindex = self.cluster.dynamical_merging_index(CoP, ParType1_CoM, apertures)
        ParType4_dynindex = self.cluster.dynamical_merging_index(CoP, ParType4_CoM, apertures)
        ParType5_dynindex = self.cluster.dynamical_merging_index(CoP, ParType5_CoM, apertures)

        data = {'/Total_dynindex'   : np.array(Total_dynindex),
                '/ParType0_dynindex': np.array(ParType0_dynindex),
                '/ParType1_dynindex': np.array(ParType1_dynindex),
                '/ParType4_dynindex': np.array(ParType4_dynindex),
                '/ParType5_dynindex': np.array(ParType5_dynindex)}

        attributes = {'Description': """Datasets with the dynamical merging index of the cluster, calculated 
                from particles within a specific aperture radius from the Centre of Potential. Individual datasets contain 
                merging index information about each particle type separately, as well as one with combined total 
                contribution.
                The dynamical merging index is computed according to the equation:
                dynamical_merging_index = || CoM(r) - CoP(r) || / r.
                
                Note: The particle type infomation combines the CoM calculated for every particle type and the 
                overall CoP of the whole FoF cluster. I.e., the CoP is not computed in a particle type-wise manner. 
                If in doubt, use the Total_dynindex dataset, which contains the dynamical merging index computed for 
                all particle types within a given aperture.
                """,
                      'Units': '[None]'}

        out = FOFOutput(self.cluster, filename='dynamical_merging_index.hdf5', data=data, attrs=attributes)
        out.makefile()

    def push_thermodynamic_merging_index(self):
        pass

    def push_substructure_merging_index(self):
        pass



if __name__ == '__main__':

    cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
    out = FOFDatagen(cluster)
    # out.push_R_crit()
    # out.push_apertures()
    # out.push_mass()
    # out.push_centre_of_mass()
    # out.push_peculiar_velocity()
    # out.push_angular_momentum()
    out.push_dynamical_merging_index()




i = 0
# @progressbar.ProgressBar()
@make_parallel_MPI
def MPI_decorator_test(**kwargs):
    import time
    nb_iter = 200
    time.sleep(0.0001)
    i=+1
    # yield ((i) / nb_iter)  # Give control back to decorator



@make_parallel_MPI
def push_FOFangmom_alignment_matrix(*args, **kwargs):
    """
    Saves the angular momentum alignment matrix data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    cluster_obj = kwargs['cluster']

    print('[ FOF SAVE ]\t==>\t AngMom align matrix on cluster {} @ z = {}'.format(cluster_obj.clusterID, cluster_obj.redshift))

    align_matrix = np.zeros((0, 6), dtype=np.float)

    # Loop over apertures
    for r in cluster_obj.generate_apertures():
        m = angular_momentum.angular_momentum_PartType_alignment_matrix(cluster_obj, specific_angular_momentum=False, aperture_radius=r)

        # Contract alignment matrix into 1D vector
        align_matrix_aperture = np.array([m[1][0], m[2][0], m[2][1], m[3][0], m[3][1], m[3][2]])
        align_matrix = np.concatenate((align_matrix, [align_matrix_aperture]), axis=0)

    assert align_matrix.__len__() == cluster_obj.generate_apertures().__len__()


    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='Group_Angular_Momentum_Alignment_Matrix',
                        input_data=align_matrix,
                        attributes="""The alignment matrix elements are calculated for each aperture listed in 
                        the 
                        `Aperture dataset`. PartTypes included: 0, 1, 4, 5.

                        Units: degrees
                        
                        Element reference:
                        0 = DM to gas
                        1 = Stars to gas
                        2 = Stars to DM
                        3 = BH to gas
                        4 = BH to DM
                        5 = BH to stars
                        """)

@make_parallel_MPI
def push_FOFmerging_indices(*args, **kwargs):
    """
    Saves the angular momentum alignment matrix data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    cluster_obj = kwargs['cluster']

    print('[ FOF SAVE ]\t==>\t Merging indices on cluster {} @ z = {}'.format(cluster_obj.clusterID, cluster_obj.redshift))

    dynamical_idx = np.zeros(0, dtype=np.float)
    thermal_idx = np.zeros(0, dtype=np.float)

    # Loop over apertures
    for r in cluster_obj.generate_apertures():
        dyn_aperture = mergers.dynamical_index(cluster_obj, aperture_radius=r)
        therm_aperture = mergers.thermal_index(cluster_obj, aperture_radius=r)

        dynamical_idx = np.concatenate((dynamical_idx, [dyn_aperture]), axis=0)
        thermal_idx = np.concatenate((thermal_idx, [therm_aperture]), axis=0)

    assert dynamical_idx.__len__() == cluster_obj.generate_apertures().__len__()
    assert thermal_idx.__len__() == cluster_obj.generate_apertures().__len__()

    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='Dynamical_Merging_Index',
                        input_data=dynamical_idx,
                        attributes="""The dynamical merging indices calculated for each aperture listed in 
                        the `Aperture dataset`. PartTypes included: 0, 1, 4, 5.

                        Units: Dimensionless
                        """)

    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='Thermal_Merging_Index',
                        input_data=thermal_idx,
                        attributes="""The thermal merging indices calculated for each aperture listed in 
                                    the `Aperture dataset`. PartTypes included: 0, 1, 4, 5.

                                    Units: Dimensionless
                                    """)



