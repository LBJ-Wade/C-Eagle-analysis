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

import numpy as np
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str


__HDF5_SUBFOLDER__ = 'FOF'

def push_FOFapertures(simulation):
    """

    :param simulation: (cluster.Simulation) object
    :return:
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

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


def push_FOFangular_momentum(simulation):
    """
    Saves the angular momentum data into the catalogues.
    :param simulation: (cluster.Simulation) object
    :return: None
    """
    simulation_obj = Simulation(simulation_name=simulation)

    for halo_num in simulation_obj.clusterIDAllowed:

        for redshift in simulation_obj.redshiftAllowed:
            cluster_obj = Cluster(clusterID=int(halo_num), redshift=redshift_str2num(redshift))
            print('[ FOF SAVE ]\t==>\t CoM on cluster {} @ z = {}'.format(halo_num, redshift))

            CoM = np.zeros((0, 3), dtype=np.float)

            for r in cluster_obj.generate_apertures():
                CoM_aperture, _ = cluster_obj.group_angular_momentum(out_allPartTypes=False, aperture_radius=r)

                # Convert into physical frame from comoving
                CoM_aperture = cluster_obj.comoving_ang_momentum(CoM_aperture)
                CoM = np.concatenate((CoM, [CoM_aperture]), axis=0)

            assert CoM.__len__() == cluster_obj.generate_apertures().__len__()


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