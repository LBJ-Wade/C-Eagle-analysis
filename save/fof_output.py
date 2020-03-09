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
import sys
import os
from mpi4py import MPI
import itertools
import numpy as np
import h5py

import save

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from _cluster_retriever import redshift_str2num, redshift_num2str
from testing import angular_momentum
from testing import mergers
import progressbar

from save.save import SimulationOutput


__HDF5_SUBFOLDER__ = 'FOF'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class FOFOutput(SimulationOutput):

    def __init__(self):
        pass
        pass

        

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
def push_FOFapertures(*args, **kwargs):

    cluster_obj = kwargs['cluster']
    print('[ FOF SAVE ]\t==>\t Apertures on cluster {} @ z = {}'.format(cluster_obj.clusterID, cluster_obj.redshift))
    save.create_dataset(kwargs['fileCompletePath'],
                       subfolder = __HDF5_SUBFOLDER__,
                       dataset_name = 'Apertures',
                       input_data = cluster_obj.generate_apertures(),
                       attributes = """Global properties of the FoF group are determined using particles
                       data, filtering particles within a specific radius from the Centre of Potential. Such
                       radius is defined as "aperture radius" and in this code is given by the method
                       cluster.Cluster.generate_apertures() in physical coordinates.
    
                       Units: Mpc
                       """)

@make_parallel_MPI
def push_FOFcentre_of_mass(*args, **kwargs):
    """
    Saves the CoM data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """

    cluster_obj = kwargs['cluster']
    print('[ FOF SAVE ]\t==>\t CoM on cluster {} @ z = {}'.format(cluster_obj.clusterID, cluster_obj.redshift))

    CoM = np.zeros((0, 3), dtype=np.float)

    # Loop over apertures
    for r in cluster_obj.generate_apertures():

        CoM_aperture, _ = cluster_obj.group_centre_of_mass(aperture_radius = r, out_allPartTypes = kwargs['out_allPartTypes'])
        CoM = np.concatenate((CoM, [CoM_aperture]), axis=0)

    assert CoM.__len__() == cluster_obj.generate_apertures().__len__()

    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='Group_Centre_of_Mass',
                        input_data= CoM,
                        attributes="""The Centre of Mass (CoM) is calculated for each aperture listed in the 
                        `Aperture dataset`. PartTypes included: 0, 1, 4, 5.

                        Units: h^-1 Mpc 
                        """)

@make_parallel_MPI
def push_FOFangular_momentum_n_mass(*args, **kwargs):
    """
    Saves the angular momentum data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    cluster_obj = kwargs['cluster']

    print('[ FOF SAVE ]\t==>\t CoM on cluster {} @ z = {}'.format(cluster_obj.clusterID, cluster_obj.redshift))

    ang_momentum = np.zeros((0, 3), dtype=np.float)
    mass = np.zeros(0, dtype=np.float)

    # Loop over apertures
    for r in cluster_obj.generate_apertures():

        ang_momentum_aperture, mass_aperture = cluster_obj.group_angular_momentum(aperture_radius = r, out_allPartTypes = kwargs['out_allPartTypes'])

        ang_momentum = np.concatenate((ang_momentum, [ang_momentum_aperture]), axis=0)
        mass = np.concatenate((mass, [mass_aperture]), axis=0)

    assert ang_momentum.__len__() == cluster_obj.generate_apertures().__len__()
    assert mass.__len__() == cluster_obj.generate_apertures().__len__()

    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='Group_Angular_Momentum',
                        input_data=ang_momentum,
                        attributes="""The total angular momentum is calculated for each aperture listed in the 
                        `Aperture dataset`. PartTypes included: 0, 1, 4, 5.

                        Units: 10^10 M_sun * 10^3 km/s * Mpc
                        """)

    save.create_dataset(kwargs['fileCompletePath'],
                        subfolder=__HDF5_SUBFOLDER__,
                        dataset_name='TotalMass',
                        input_data=mass,
                        attributes="""The total mass is calculated for each aperture listed in the 
                                    `Aperture dataset. PartTypes included: 0, 1, 4, 5.

                                    Units: 10^10 M_sun
                                    """)

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



