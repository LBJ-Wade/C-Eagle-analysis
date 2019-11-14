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
from memory import *



class Mixin:

    @staticmethod
    def centre_of_mass(mass, coords):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        ACCESS DATA: e.g. group_CoM[0] for getting the x value
        """
        assert mass.__len__() > 0, "Array is empty."
        assert coords.__len__() > 0, "Array is empty."
        assert mass.__len__() == coords.__len__(), "Mass and coords arrays do not have same size."

        sum_of_masses = np.sum(mass)
        centre_of_mass = np.average(coords, axis=0, weights=np.divide(mass, sum_of_masses))
        free_memory(['centre_of_mass', 'sum_of_masses'], invert=True)
        return centre_of_mass, sum_of_masses

    @staticmethod
    def zero_momentum_frame(mass, vel):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        """
        assert mass.__len__() > 0, "Array is empty."
        assert vel.__len__() > 0, "Array is empty."
        assert mass.__len__() == vel.__len__(), "Mass and vel arrays do not have same size."

        sum_of_masses = np.sum(mass)
        zero_momentum = np.average(vel, axis=0, weights=np.divide(mass, sum_of_masses))
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

            # Import data
            mass = self.particle_masses(part_type)
            coords = self.particle_coordinates(part_type)
            group_num = self.group_number_part(part_type)

            # Filter local distribution of matter (r<R500)
            r500 = self.group_r500()
            group_CoP = self.group_centre_of_potential()
            coords = np.subtract(coords, group_CoP)
            r = np.linalg.norm(coords, axis=1)
            index = np.where((r < r500) & (group_num == 1))[0]
            mass = mass[index]
            coords = coords[index]
            assert mass.__len__() > 0, "Array is empty - check filtering."
            assert coords.__len__() > 0, "Array is empty - check filtering."

            # Compute CoM for each particle type
            centre_of_mass, sum_of_masses = self.centre_of_mass(mass, coords)
            CoM_PartTypes = np.append(CoM_PartTypes, [centre_of_mass], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

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

            # Import data
            mass = self.particle_masses(part_type)
            vel = self.particle_velocity(part_type)
            coords = self.particle_coordinates(part_type)
            group_num = self.group_number_part(part_type)

            # Filter local distribution of matter (r<R500)
            r500 = self.group_r500()
            group_CoP = self.group_centre_of_potential()
            coords = np.subtract(coords, group_CoP)
            r = np.linalg.norm(coords, axis=1)
            index = np.where((r < r500) & (group_num == 1))[0]
            mass = mass[index]
            vel = vel[index]
            assert mass.__len__() > 0, "Array is empty - check filtering.."
            assert vel.__len__() > 0, "Array is empty - check filtering."

            # Compute *local* ZMF for each particle type
            zero_momentum, sum_of_masses = self.zero_momentum_frame(mass, vel)
            ZMF_PartTypes = np.append(ZMF_PartTypes, [zero_momentum], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

        if out_allPartTypes:
            return ZMF_PartTypes, Mtot_PartTypes
        else:
            return self.zero_momentum_frame(Mtot_PartTypes, ZMF_PartTypes)

    @staticmethod
    def kinetic_energy(mass, vel):
        ke = 0.5 * mass * np.linalg.norm(vel, axis = 1)**2
        return np.sum(ke)

    @staticmethod
    def thermal_energy(mass, temperature):
        k_B = 1.38064852 * np.power(10, -23.)
        te = 1.5 * k_B * temperature * mass * 0.88 / (1.6735575* np.power(10, -27.))
        return np.sum(te)

    # @staticmethod
    # def ang_momentum(mass, vel, coords, origin):
    #     r = func(coords, rot_axis)
    #     angmom_part = np.outer(r, np.multiply(mass, vel))
    #     return np.sum(angmom_part)

    #####################################################
    #													#
    #				COMOVING UNITS      				#
    # 									 				#
    #####################################################


    def comoving_density(self, density):
        """
        Rescales the density from the comoving coordinates to the physical coordinates
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(density,  hubble_par ** 2 * scale_factor ** -3)

    def comoving_length(self, coord):
        """
        Rescales the density from the comoving length to the physical length
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(coord, scale_factor / hubble_par)

    def comoving_velocity(self, vel):
        """
        Rescales the density from the comoving velocity to the physical velocity
        """
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(vel, np.sqrt(scale_factor))

    def comoving_mass(self, mass):
        """
        Rescales the density from the comoving mass to the physical mass
        """
        hubble_par = self.file_hubble_param()
        return np.divide(mass, hubble_par)

    def comoving_momentum(self, mom):
        """
        Rescales the momentum from the comoving to the physical
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(mom, np.sqrt(scale_factor) / hubble_par)


    #####################################################
    #													#
    #			    UNITS CONEVRSION     				#
    # 									 				#
    #####################################################

    @staticmethod
    def density_units(density, unit_system='SI'):
        """
        CREATED: 12.02.2019
        LAST MODIFIED: 12.02.2019

        INPUTS: density np.array

                metric system used: 'SI' or 'cgs' or astronomical 'astro'
        """
        if unit_system == 'SI':
            # kg*m^-3
            conv_factor = 6.769911178294543 * 10 ** -28
        elif unit_system == 'cgs':
            # g*cm^-3
            conv_factor = 6.769911178294543 * 10 ** -31
        elif unit_system == 'astro':
            # solar masses / (parsec)^3
            conv_factor = 6.769911178294543 * np.power(3.086, 3.) / 1.9891 * 10 ** -10
        elif unit_system == 'nHcgs':
            conv_factor = 6.769911178294543 * 10 ** -31 / (1.674 * 10 ** -24.)
        else:
            raise("[ERROR] Trying to convert SPH density to an unknown metric system.")

        return np.multiply(density, conv_factor)

    @staticmethod
    def velocity_units(velocity, unit_system='SI'):
        """
        CREATED: 14.02.2019
        LAST MODIFIED: 14.02.2019

        INPUTS: velocity np.array

                metric system used: 'SI' or 'cgs' or astronomical 'astro'
        """
        if unit_system == 'SI':
            # m/s
            conv_factor =  10 ** 3
        elif unit_system == 'cgs':
            # cm/s
            conv_factor =  10 ** 5
        elif unit_system == 'astro':
            # km/s
            conv_factor =  1
        else:
            raise("[ERROR] Trying to convert velocity to an unknown metric system.")

        return np.multiply(velocity, conv_factor)



    @staticmethod
    def mass_units(mass, unit_system='SI'):
        """
        CREATED: 14.02.2019
        LAST MODIFIED: 14.02.2019

        INPUTS: mass np.array

                metric system used: 'SI' or 'cgs' or astronomical 'astro'
        """
        if unit_system == 'SI':
            # m/s
            conv_factor = 1.9891 * 10 ** 40
        elif unit_system == 'cgs':
            # cm/s
            conv_factor = 1.9891 * 10 ** 43
        elif unit_system == 'astro':
            # km/s
            conv_factor = 10 ** 10
        else:
            raise("[ERROR] Trying to convert mass to an unknown metric system.")

        return np.multiply(mass, conv_factor)

    @staticmethod
    def momentum_units(momentum, unit_system='SI'):
        """
        CREATED: 07.03.2019
        LAST MODIFIED: 07.03.2019

        INPUTS: momentum np.array

                metric system used: 'SI' or 'cgs' or astronomical 'astro'
        """
        if unit_system == 'SI':
            # m/s
            conv_factor = 1.9891 * 10 ** 43
        elif unit_system == 'cgs':
            # cm/s
            conv_factor = 1.9891 * 10 ** 48
        elif unit_system == 'astro':
            # km/s
            conv_factor = 10 ** 10
        else:
            raise("[ERROR] Trying to convert mass to an unknown metric system.")

        return np.multiply(momentum, conv_factor)