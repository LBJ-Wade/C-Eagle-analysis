"""
------------------------------------------------------------------
FILE:   _cluster_profiler.py
AUTHOR: Edo Altamura
DATE:   12-11-2019
------------------------------------------------------------------
This file is an extension of the cluster.Cluster class. It provides
class methods for performing basic analysis procedures on C-EAGLE
data from the /cosma5 data system.
This file contains a mixin class, affiliated to cluster.Cluster.
Mixins are classes that have no data of their own — only methods —
so although you inherit them, you never have to call super() on them.
They working principle is based on OOP class inheritance.
-------------------------------------------------------------------
"""

from cluster import *


class Mixin():

    @staticmethod
    def centre_of_mass(mass, coords):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        ACCESS DATA: e.g. group_CoM[0] for getting the x value
        """
        sum_of_masses = np.sum(mass)
        centre_of_mass = np.array([np.sum(mass * coords[:, 0]) / sum_of_masses,
                                   np.sum(mass * coords[:, 1]) / sum_of_masses,
                                   np.sum(mass * coords[:, 2]) / sum_of_masses])
        free_memory(['centre_of_mass', 'sum_of_masses'], invert=True)
        return centre_of_mass, sum_of_masses

    @staticmethod
    def zero_momentum_frame(mass, vel):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        """
        sum_of_masses = np.sum(mass)
        zero_momentum = np.array([np.sum(mass * vel[:, 0]) / sum_of_masses,
                                  np.sum(mass * vel[:, 1]) / sum_of_masses,
                                  np.sum(mass * vel[:, 2]) / sum_of_masses])
        free_memory(['zero_momentum', 'sum_of_masses'], invert=True)
        return zero_momentum, sum_of_masses

    def group_centre_of_mass(self, out_allPartTypes=False):
        """
        out_allPartTypes = (bool)
            if True outputs the centre of mass and sum of masses of each
            partType separately in arrays

            if False outputs the overall CoM and sum of masses of the whole
            cluster.

        Returns the centre of mass of the cluster for a ALL particle types,
        except for lowres_DM (2, 3).
        """
        CoM_PartTypes = np.zeros((0, 3), dtype=np.float)
        Mtot_PartTypes = np.zeros(0, dtype=np.float)

        for part_type in ['0', '1', '4', '5']:
            mass = self.particle_masses(part_type)
            coords = self.particle_coordinates(part_type)
            centre_of_mass, sum_of_masses = self.centre_of_mass(mass, coords)
            CoM_PartTypes = np.append(CoM_PartTypes, [centre_of_mass], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)
            free_memory(['CoM_PartTypes', 'Mtot_PartTypes'], invert=True)

        if out_allPartTypes:
            return CoM_PartTypes, Mtot_PartTypes
        else:
            return self.centre_of_mass(Mtot_PartTypes, CoM_PartTypes)

    def group_zero_momentum_frame(self, out_allPartTypes=False):
        """
        out_allPartTypes = (bool)
            if True outputs the zero_momentum_frame and sum of masses of each
            partType separately in arrays

            if False outputs the overall zero_momentum_frame and sum of masses
            of the whole cluster.

        Returns the zero_momentum_frame of the cluster for a ALL particle types,
        except for lowres_DM (2, 3).
        """
        ZMF_PartTypes = np.zeros((0, 3), dtype=np.float)
        Mtot_PartTypes = np.zeros(0, dtype=np.float)

        for part_type in ['0', '1', '4', '5']:
            mass = self.particle_masses(part_type)
            vel = self.particle_velocity(part_type)
            zero_momentum, sum_of_masses = self.zero_momentum_frame(mass, vel)
            ZMF_PartTypes = np.append(ZMF_PartTypes, [zero_momentum], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)
            free_memory(['ZMF_PartTypes', 'Mtot_PartTypes'], invert=True)

        if out_allPartTypes:
            return ZMF_PartTypes, Mtot_PartTypes
        else:
            return self.zero_momentum_frame(Mtot_PartTypes, ZMF_PartTypes)


    @staticmethod
    def kinetic_energy(mass, vel):
        ke = 0.5 * mass * np.sum([vel[i] ** 2 for i in range(0, 3)])
        return np.sum(ke)

    @staticmethod
    def thermal_energy(mass, temperature):
        k_B = 1.38064852e-23
        te = 1.5 * k_B * temperature * mass * 0.88 / (1.6735575e-27)
        return np.sum(te)