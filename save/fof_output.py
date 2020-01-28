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

from save import save

from mpi4py import MPI
import itertools
import numpy as np
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str
from testing import angular_momentum
from testing import mergers


__HDF5_SUBFOLDER__ = 'FOF'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def push_FOFapertures(simulation):
    """

    :param simulation: (cluster.Simulation) object
    :return:
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        # Allocate jobs to MPI interface
        if halo_num % size != rank:
            continue

        for redshift in simulation_obj.redshiftAllowed:

            cluster_obj = Cluster(clusterID = int(halo_num), redshift = redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t Apertures on cluster {} @ z = {}'.format(halo_num, redshift))
            save.create_dataset(simulation,
                               cluster_obj,
                               subfolder = __HDF5_SUBFOLDER__,
                               dataset_name = 'Apertures',
                               input_data = cluster_obj.generate_apertures(),
                               attributes = """Global properties of the FoF group are determined using particles
                               data, filtering particles within a specific radius from the Centre of Potential. Such
                               radius is defined as "aperture radius" and in this code is given by the method
                               cluster.Cluster.generate_apertures() in physical coordinates.

                               Units: Mpc
                               """,
                                )


def push_FOFcentre_of_mass(simulation):
    """
    Saves the CoM data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        # Allocate jobs to MPI interface
        if halo_num % size != rank:
            continue

        for redshift in simulation_obj.redshiftAllowed:
            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t CoM on cluster {} @ z = {}'.format(halo_num, redshift))

            CoM = np.zeros((0, 3), dtype=np.float)

            for r in cluster_obj.generate_apertures(comoving = True):
                CoM_aperture, _ = cluster_obj.group_centre_of_mass(
                    out_allPartTypes=False, aperture_radius=r)

                CoM = np.concatenate((CoM, [CoM_aperture]), axis=0)

            assert CoM.__len__() == cluster_obj.generate_apertures().__len__()

            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='Group_Centre_of_Mass',
                                input_data= CoM,
                                attributes="""The Centre of Mass (CoM) is calculated for each aperture listed in the 
                                "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                Units: h^-1 Mpc 
                                """,
                                )


def push_FOFangular_momentum_n_masse(simulation):
    """
    Saves the angular momentum data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        # Allocate jobs to MPI interface
        if halo_num % size != rank:
            continue

        for redshift in simulation_obj.redshiftAllowed:
            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t CoM on cluster {} @ z = {}'.format(halo_num, redshift))

            CoM = np.zeros((0, 3), dtype=np.float)
            mass = np.zeros(0, dtype=np.float)

            for r in cluster_obj.generate_apertures():
                CoM_aperture, mass_aperture = cluster_obj.group_angular_momentum(out_allPartTypes=False,
                                                                               aperture_radius=r)

                # Convert into physical frame from comoving
                CoM_aperture = cluster_obj.comoving_ang_momentum(CoM_aperture)
                CoM = np.concatenate((CoM, [CoM_aperture]), axis=0)

                # Convert into physical frame from comoving
                mass_aperture = cluster_obj.comoving_mass(mass_aperture)
                mass = np.concatenate((mass, [mass_aperture]), axis=0)

            assert CoM.__len__() == cluster_obj.generate_apertures().__len__()
            assert mass.__len__() == cluster_obj.generate_apertures().__len__()


            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='Group_Angular_Momentum',
                                input_data=CoM,
                                attributes="""The total angular momentum is calculated for each aperture listed in the 
                                "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                Units: 10^10 M_sun * 10^3 km/s * Mpc
                                """,
                                )

            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='TotalMass',
                                input_data=mass,
                                attributes="""The total mass is calculated for each aperture listed in the 
                                            "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                            Units: 10^10 M_sun
                                            """,
                                )


def push_FOFangmom_alignment_matrix(simulation):
    """
    Saves the angular momentum alignment matrix data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        # Allocate jobs to MPI interface
        if halo_num % size != rank:
            continue

        for redshift in simulation_obj.redshiftAllowed:
            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t AngMom align matrix on cluster {} @ z = {}'.format(halo_num, redshift))

            align_matrix = np.zeros((0, 6), dtype=np.float)

            for r in cluster_obj.generate_apertures():
                m = angular_momentum.angular_momentum_PartType_alignment_matrix(cluster_obj,
                                                                                                    specific_angular_momentum=False,
                                                               aperture_radius=r)

                # Contract alignment matrix into 1D vector
                align_matrix_aperture = np.array([m[1][0], m[2][0], m[2][1], m[3][0], m[3][1], m[3][2]])

                align_matrix = np.concatenate((align_matrix, [align_matrix_aperture]), axis=0)

            assert align_matrix.__len__() == cluster_obj.generate_apertures().__len__()


            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='Group_Angular_Momentum_Alignment_Matrix',
                                input_data=align_matrix,
                                attributes="""The alignment matrix elements are calculated for each aperture listed in 
                                the 
                                "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                Units: degrees
                                
                                Element reference:
                                0 = DM to gas
                                1 = Stars to gas
                                2 = Stars to DM
                                3 = BH to gas
                                4 = BH to DM
                                5 = BH to stars
                                """,
                                )


def push_FOFmerging_indices(simulation):
    """
    Saves the angular momentum alignment matrix data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        # Allocate jobs to MPI interface
        if halo_num % size != rank:
            continue

        for redshift in simulation_obj.redshiftAllowed:
            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t Merging indices on cluster {} @ z = {}'.format(halo_num, redshift))

            dynamical_idx = np.zeros(0, dtype=np.float)
            thermal_idx = np.zeros(0, dtype=np.float)

            for r in cluster_obj.generate_apertures():
                dyn_aperture = mergers.dynamical_index(cluster_obj, aperture_radius=r)
                therm_aperture = mergers.thermal_index(cluster_obj, aperture_radius=r)

                dynamical_idx = np.concatenate((dynamical_idx, [dyn_aperture]), axis=0)
                thermal_idx = np.concatenate((thermal_idx, [therm_aperture]), axis=0)

            assert dynamical_idx.__len__() == cluster_obj.generate_apertures().__len__()
            assert thermal_idx.__len__() == cluster_obj.generate_apertures().__len__()

            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='Dynamical_Merging_Index',
                                input_data=dynamical_idx,
                                attributes="""The dynamical merging indices calculated for each aperture listed in 
                                the 
                                "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                Units: Dimensionless

                                """,
                                )

            save.create_dataset(simulation,
                                cluster_obj,
                                subfolder=__HDF5_SUBFOLDER__,
                                dataset_name='Thermal_Merging_Index',
                                input_data=thermal_idx,
                                attributes="""The thermal merging indices calculated for each aperture listed in 
                                            the 
                                            "Aperture dataset". PartTypes included: 0, 1, 4, 5.

                                            Units: Dimensionless

                                            """,
                                )


def push_FOFtest_MPI(simulation):
    """
    Saves the angular momentum alignment matrix data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    sim = Simulation(simulation_name=simulation)

    process = 0
    process_iterator = itertools.product(sim.clusterIDAllowed, sim.redshiftAllowed)

    for halo_num, redshift in process_iterator:

        if process % size == rank:

            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))

            print('Processor ({}/{}) is processing halo ({} @ {})'.format(rank, size, cluster_obj.clusterID,
                                                                          cluster_obj.redshift))

            # for r in cluster_obj.generate_apertures():
            #     pass

        process += 1
