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

import numpy as np
from typing import List, Dict, Tuple
from unyt import hydrogen_mass, boltzmann_constant, gravitational_constant, parsec, solar_mass
from .memory import free_memory
import warnings

# Delete the units from Unyt constants
hydrogen_mass = float(hydrogen_mass.value)
boltzmann_constant = float(boltzmann_constant.value)
gravitational_constant = float(gravitational_constant.value)
parsec = float((1*parsec).in_units('m').value)
solar_mass = float(solar_mass.value)

class Mixin:

    @staticmethod
    def angle_between_vectors(v1, v2):
        # v1 is your firsr vector
        # v2 is your second vector

        v1 = np.array(v1)
        v2 = np.array(v2)

        # If the vectors are almost equal, then the arccos returns nan because of numerical
        # precision: catch that case and return 0 angle instead.
        if np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) - 1) < 1e-14:
            # print(f"[ Angle between vectors]\t==> Warning: the vectors {v1} and {v2} are very well aligned to better "
            #       f"than 1e-12. The separation angle has therefore been set to 0. deg")
            angle = 0.
        elif np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1) < 1e-14:
            # print(f"[ Angle between vectors]\t==> Warning: the vectors {v1} and {v2} are very well anti-aligned to "
            #       f"better "
            #       f"than 1e-12. The separation angle has therefore been set to 180. deg")
            angle = np.pi
        else:
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # Return the result in degrees
        return angle * 180 / np.pi

    def radial_distance_CoP(self, coords):
        coords = np.asarray(coords)
        coordinates_radial = np.sqrt(
            (coords[:, 0] - self.centre_of_potential[0]) ** 2 +
            (coords[:, 1] - self.centre_of_potential[1]) ** 2 +
            (coords[:, 2] - self.centre_of_potential[2]) ** 2
        )
        return coordinates_radial

    @staticmethod
    def kinetic_energy(mass, vel):
        mass = np.asarray(mass)
        vel = np.asarray(vel)
        ke = 0.5 * mass * np.linalg.norm(vel, axis = 1)**2
        return np.sum(ke)

    @staticmethod
    def thermal_energy(mass, temperature):
        mass = np.asarray(mass)
        temperature = np.asarray(temperature)
        te = 1.5 * boltzmann_constant * temperature * mass / (hydrogen_mass / 1.16)
        return np.sum(te)

    @staticmethod
    def centre_of_mass(mass, coords):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        ACCESS DATA: e.g. group_CoM[0] for getting the x value
        """
        mass   = np.asarray(mass)
        coords = np.asarray(coords)
        return np.sum(coords*mass[:, None], axis = 0)/np.sum(mass)

    @staticmethod
    def zero_momentum_frame(mass, velocity):
        """
        AIM: reads the FoF group central of mass from the path and file given
        RETURNS: type = np.array of 3 doubles
        """
        mass     = np.asarray(mass)
        velocity = np.asarray(velocity)
        return np.sum(velocity*mass[:, None], axis = 0)/np.sum(mass)

    @staticmethod
    def angular_momentum(mass, coords, velocity):
        """
        Compute angular momentum as r [cross] mv.

        :param mass: the mass array of the particles
        :param coords: the coordinates of the particles, rescaled to the origin of reference
            Usually, coords are rescaled with respect to the centre of potential.
        :param velocity:  the velocity array of the particles, rescaled to the cluster's
            rest frame. I/e/ take out the bulk peculiar velocity to isolate the rotation.
        :return: np.array with the 3D components of the angular momentum vector.
        """
        mass = np.asarray(mass)
        coords = np.asarray(coords)
        velocity = np.asarray(velocity)
        linear_momentum_r = velocity * mass[:, None]
        return np.sum(np.cross(coords, linear_momentum_r), axis=0) / np.sum(mass)

    @staticmethod
    def inertia_tensor(mass: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
		Compute the moment of inertia tensor in matrix form:

		            I_xx    I_xy    I_xz
		            I_yx    I_yy    I_yz
		            I_zx    I_zy    I_zz

		:param mass: the mass array of the particles
		:param coords: the coordinates of the particles, rescaled to the origin of reference
			Usually, coords are rescaled with respect to the centre of potential.

		:return: np.array with the 3x3 component inertia tensor.
		"""
        m = np.asarray(mass)
        coords = np.asarray(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        del coords
        I_xx = np.sum(m*(y**2+z**2))
        I_xy = - np.sum(m*x*y)
        I_xz = - np.sum(m*x*z)
        I_yx = I_xy
        I_yy = np.sum(m*(x**2+z**2))
        I_yz = - np.sum(m*y*z)
        I_zx = I_xz
        I_zy = I_yz
        I_zz = np.sum(m*(x**2+y**2))
        return np.array([
                [I_xx, I_xy, I_xz],
                [I_yx, I_yy, I_yz],
                [I_zx, I_zy, I_zz],
        ])

    @staticmethod
    def principal_axes_ellipsoid(inertia_tensor: np.ndarray, eigenvalues: bool = False) -> np.ndarray:
        """
        Compute the eigenvalues and eigenvectors of the inertia tensor for morphology studies
        :param inertia_tensor:
        :return:
        """
        eigensolution = np.linalg.eig(inertia_tensor)
        return eigensolution if eigenvalues else eigensolution[1]

    def generate_apertures(self):
        """
        Generate an array of apertures for calculating global properties of the clusters.
        The apertures use both R2500 and R200 units:

        :return: (np.ndarray)
            The array with 100 different apertures, ranging from ~ R2500 to 5*R200
            NOTE: the apertures are returned in the PHYSICAL frame.
        """
        assert self.r2500 > 0. and self.r200 > 0., f"Issue encountered at {self.clusterID}, {self.redshift}"
        return np.logspace(np.log10(self.r200 / 4), np.log10(5 * self.r200), 20)


    def group_thermal_energy(self,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the thermal_energy of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the total thermal_energy.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall thermal_energy, or return the of each particle type separately.
        This toggle is controlled by a boolean.

        The thermal_energy is computed after converting the datasets from Gadget units to
        SI units ([mass] = kg, [velocity] = km/s). The output is given in units of
        (10**46) Joules.

        thermal_energy = 3/2 * k_B * T * (number of H atoms)
        number of H atoms = mass

        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        part_type = '0'
        assert hasattr(self, f'partType{part_type}_coordinates')
        assert hasattr(self, f'partType{part_type}_temperature')
        assert hasattr(self, f'partType{part_type}_mass')
        radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
        aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
        free_memory(['radial_dist'])
        mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
        temperature = getattr(self, f'partType{part_type}_temperature')[aperture_radius_index]
        if mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
        if temperature.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
        mass = self.mass_units(mass, unit_system='SI')
        return self.thermal_energy(mass, temperature)*np.power(10., -46)
    
    def group_kinetic_energy(self,
                             out_allPartTypes: bool =False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the kinetic_energy of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the total kinetic_energy.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall kinetic_energy, or return the of each particle type separately.
        This toggle is controlled by a boolean.

        The kinetic_energy is computed after converting the datasets from Gadget units to
        SI units ([mass] = kg, [velocity] = km/s). The output is given in units of
        (10**46) Joules.

        kinetic_energy = 1/2 * mass * || velocity ||^2.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            bulk_velocity = self.group_zero_momentum_frame(aperture_radius=aperture_radius)
            kinetic_energy_PartTypes = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                _velocity = np.subtract(_velocity, bulk_velocity)
                _mass     = self.mass_units(_mass, unit_system='SI')
                _velocity = self.velocity_units(_velocity, unit_system='SI')
                kinetic_energy = self.kinetic_energy(_mass, _velocity)
                kinetic_energy_PartTypes = np.append(kinetic_energy_PartTypes, kinetic_energy)

            return kinetic_energy_PartTypes*np.power(10., -46)

        else:
            bulk_velocity = self.group_zero_momentum_frame(aperture_radius=aperture_radius)
            mass = np.zeros(0, dtype=np.float)
            velocity = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                mass = np.concatenate((mass, _mass), axis=0)
                velocity = np.concatenate((velocity, _velocity), axis=0)

            velocity = np.subtract(velocity, bulk_velocity)
            mass     = self.mass_units(mass, unit_system='SI')
            velocity = self.velocity_units(velocity, unit_system='SI')
            return self.kinetic_energy(mass, velocity)*np.power(10., -46)

    def group_mass_aperture(self,
                             out_allPartTypes: bool =False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the mass of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the total particle mass.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall mass, ot return the of each particle type separately.
        This toggle is controlled by a boolean.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            mass_PartTypes = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                sum_of_masses = np.sum(_mass)
                mass_PartTypes = np.append(mass_PartTypes, sum_of_masses)

            return mass_PartTypes

        else:

            mass   = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                mass = np.append(mass, _mass)

            return np.sum(mass)

    def group_substructure_mass(self,
                             out_allPartTypes: bool =False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the mass of particles within a specified aperture and belonging to
        subgroups.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the total particle subgroups mass.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall substructure mass, ot return the of each particle type separately.
        This toggle is controlled by a boolean.

        substructure_mass = total mass - fuzz mass

        N.B.: The Fuzz mass is given by subgroupnumber = 0.
        N.B.: The aperture radius could cut subgroups at the outer edge.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            total_mass = self.group_mass_aperture(out_allPartTypes=True, aperture_radius=aperture_radius)
            substructure_mass_PartTypes = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_subgroupnumber')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')
                aperture_radius_index = np.where((radial_dist < aperture_radius) & (subgroupnumber == 0))[0]
                free_memory(['radial_dist', 'subgroupnumber'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                fuzz_mass = np.sum(_mass)
                substructure_mass = total_mass[['0', '1', '4'].index(part_type)] - fuzz_mass
                substructure_mass_PartTypes = np.append(substructure_mass_PartTypes, substructure_mass)

            return substructure_mass_PartTypes

        else:

            total_mass = self.group_mass_aperture(out_allPartTypes=False, aperture_radius=aperture_radius)
            fuzz_mass = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_subgroupnumber')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')
                aperture_radius_index = np.where((radial_dist < aperture_radius) & (subgroupnumber == 0))[0]
                free_memory(['radial_dist', 'subgroupnumber'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                fuzz_mass = np.append(fuzz_mass, _mass)

            substructure_mass = total_mass - np.sum(fuzz_mass)
            return substructure_mass

    def group_substructure_fraction(self,
                             out_allPartTypes: bool =False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the fraction of mass constituting substructures, taking particles
        within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the total particle substructure fraction.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall substructure mass fraction, ot return the of each particle type separately.
        This toggle is controlled by a boolean.

        substructure_fraction = (total mass - fuzz mass) / total mass

        N.B.: The Fuzz mass is given by subgroupnumber = 0.
        N.B.: The aperture radius could cut subgroups at the outer edge.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            total_mass = self.group_mass_aperture(out_allPartTypes=True, aperture_radius=aperture_radius)
            substructure_frac_PartTypes = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_subgroupnumber')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')
                aperture_radius_index = np.where((radial_dist < aperture_radius) & (subgroupnumber == 0))[0]
                free_memory(['radial_dist', 'subgroupnumber'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                fuzz_mass = np.sum(_mass)
                substructure_mass = 1 - (fuzz_mass/total_mass[['0', '1', '4', '5'].index(part_type)])
                substructure_frac_PartTypes = np.append(substructure_frac_PartTypes, substructure_mass)

            return substructure_frac_PartTypes

        else:

            total_mass = self.group_mass_aperture(out_allPartTypes=False, aperture_radius=aperture_radius)
            fuzz_mass = np.zeros(0, dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_subgroupnumber')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')
                aperture_radius_index = np.where((radial_dist < aperture_radius) & (subgroupnumber == 0))[0]
                free_memory(['radial_dist', 'subgroupnumber'])
                _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                fuzz_mass = np.append(fuzz_mass, _mass)

            substructure_fraction = 1 - (np.sum(fuzz_mass)/total_mass)
            return substructure_fraction

    def group_centre_of_mass(self,
                             out_allPartTypes: bool =False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the centre of mass of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the centre of mass using the relative
        static method in this Mixin class.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall centre of mass, ot return the centre of mass of each particle type separately.
        This toggle is controlled by a boolean.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            CoM_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass   = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                centre_of_mass = self.centre_of_mass(_mass, _coords)
                CoM_PartTypes = np.concatenate((CoM_PartTypes, [centre_of_mass]), axis=0)

            return CoM_PartTypes

        else:

            mass   = np.zeros(0, dtype=np.float)
            coords = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass   = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                mass   = np.concatenate((mass, _mass), axis=0)
                coords = np.concatenate((coords, _coords), axis=0)

            return self.centre_of_mass(mass, coords)

    def group_dynamical_merging_index(self,
                             out_allPartTypes: bool = False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the group_dynamical_merging_index of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the merging index using the relative
        static method in this Mixin class.

        group_dynamical_merging_index = || CoM(aperture) - CoP || / aperture

        It also has the option of combining all the particles of different types and computing
        the overall centre of mass, ot return the centre of mass of each particle type separately.
        This toggle is controlled by a boolean.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        centre_of_mass = self.group_centre_of_mass(out_allPartTypes=out_allPartTypes,
                                                   aperture_radius=aperture_radius)
        if out_allPartTypes:
            dynamical_merging_index = self.radial_distance_CoP(centre_of_mass)/aperture_radius
        else:
            dynamical_merging_index = self.radial_distance_CoP(centre_of_mass[:, None].T)/aperture_radius
            dynamical_merging_index = dynamical_merging_index[0]

        return dynamical_merging_index

    def group_zero_momentum_frame(self,
                             out_allPartTypes: bool = False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the cluster's rest frame of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the rest frame using the relative
        static method in this Mixin class.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall bulk velocity, ot return the bulk velocity of each particle type separately.
        This toggle is controlled by a boolean.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            ZMF_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                zmf = self.zero_momentum_frame(_mass, _velocity)
                ZMF_PartTypes = np.concatenate((ZMF_PartTypes, [zmf]), axis=0)

            return ZMF_PartTypes

        else:

            mass     = np.zeros(0, dtype=np.float)
            velocity = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                mass     = np.concatenate((mass, _mass), axis=0)
                velocity = np.concatenate((velocity, _velocity), axis=0)

            return self.zero_momentum_frame(mass, velocity)


    def group_angular_momentum(self,
                             out_allPartTypes: bool = False,
                             aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the cluster's angular momentum from particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the angular momentum using the relative
        static method in this Mixin class.
        The method also checks that the necessary datasets are loaded into the cluster object.
        It also has the option of combining all the particles of different types and computing
        the overall angular momentum, or return the angular momentum of each particle type separately.
        This toggle is controlled by a boolean.

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        if out_allPartTypes:

            ANG_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords   = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

                # Rescale coordinates and velocity
                _coords   = np.subtract(_coords, self.centre_of_potential)
                _velocity = np.subtract(_velocity, self.group_zero_momentum_frame(aperture_radius=aperture_radius))
                ang = self.angular_momentum(_mass, _coords, _velocity)
                ANG_PartTypes = np.concatenate((ANG_PartTypes, [ang]), axis=0)

            return ANG_PartTypes

        else:

            mass     = np.zeros(0, dtype=np.float)
            coords   = np.zeros((0, 3), dtype=np.float)
            velocity = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords   = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
                mass     = np.concatenate((mass, _mass), axis=0)
                coords   = np.concatenate((coords, _coords), axis=0)
                velocity = np.concatenate((velocity, _velocity), axis=0)

            # Rescale coordinates and velocity
            coords   = np.subtract(coords, self.centre_of_potential)
            velocity = np.subtract(velocity, self.group_zero_momentum_frame(aperture_radius=aperture_radius))
            return self.angular_momentum(mass, coords, velocity)

    def group_thermodynamic_merging_index(self,
                                      aperture_radius: float = None) -> np.ndarray:
        """
        Method that computes the group_thermodynamic_merging_index of particles within a specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the merging index using the relative
        static method in this Mixin class.

        group_thermodynamic_merging_index = kinetic_energy / thermal_energy

        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        kinetic_energy = self.group_kinetic_energy(out_allPartTypes=True, aperture_radius=aperture_radius)[0]
        thermal_energy = self.group_thermal_energy(aperture_radius=aperture_radius)
        return kinetic_energy / thermal_energy

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """
        Find the rotation matrix that aligns vec1 to vec2

        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        # Catch exception where the vectors are perfectly aligned
        if Mixin.angle_between_vectors(vec1, vec2) < 1e-10:
            return np.identity(3)
        elif np.abs(Mixin.angle_between_vectors(vec1, vec2) - 180) < 1e-10:
            return (-1) * np.identity(3)
        else:
            return np.asarray(rotation_matrix)

    @staticmethod
    def rotation_matrix_about_axis(axis, theta):
        """
        Find the rotation matrix that rotates a vector about the origin by delta(theta) in the elevation
        and by delta(phi) about the azimuthal axis.

        :param theta: expect float in degrees
            The displacement angle in elevation.

        :param phi: expect float in degrees
            The displacement angle in azimuth.

        :return: A transform matrix (3x3)
        """
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


    def rotation_matrix_about_x(self, angle):
        axis = [1,0,0]
        return self.rotation_matrix_about_axis(axis, angle)

    def rotation_matrix_about_y(self, angle):
        axis = [0,1,0]
        return self.rotation_matrix_about_axis(axis, angle)

    def rotation_matrix_about_z(self, angle):
        axis = [0,0,1]
        return self.rotation_matrix_about_axis(axis, angle)

    @staticmethod
    def _TEST_apply_rotation_matrix(rot_matrix, vectors):
        """Apply this rotation to a set of vectors.
        This function is a self-made test method for comparison with the `scipy`
        method implemented below.

        :param rot_matrix: array_like, shape (3,3) or (N, 3, 3)
            Each matrix represents a rotation in 3D space of the corresponding
            vector.

        :param vectors: array_like, shape (3,) or (N, 3)
            Each `vectors[i]` represents a vector in 3D space. A single vector
            can either be specified with shape `(3, )` or `(1, 3)`. The number
            of rotations and number of vectors given must follow standard numpy
            broadcasting rules: either one of them equals unity or they both
            equal each other.

        :return rotated_vectors : ndarray, shape (3,) or (N, 3)
            Result of applying rotation on input vectors.
            Shape depends on the following cases:
                - If object contains a single rotation (as opposed to a stack
                  with a single rotation) and a single vector is specified with
                  shape ``(3,)``, then `rotated_vectors` has shape ``(3,)``.
                - In all other cases, `rotated_vectors` has shape ``(N, 3)``,
                  where ``N`` is either the number of rotations or vectors.
        """

        vectors = np.asarray(vectors)
        rot_matrix = np.asarray(rot_matrix)

        assert rot_matrix.shape == (3,3)
        assert vectors.__len__() > 0
        return rot_matrix.dot(vectors)

    @staticmethod
    def apply_rotation_matrix(rot_matrix, vectors, inverse=False):
        """Apply this rotation to a set of vectors.
        If the original frame rotates to the final frame by this rotation, then
        its application to a vector can be seen in two ways:
            - As a projection of vector components expressed in the final frame
              to the original frame.
            - As the physical rotation of a vector being glued to the original
              frame as it rotates. In this case the vector components are
              expressed in the original frame before and after the rotation.
        In terms of rotation matricies, this application is the same as
        ``self.as_matrix().dot(vectors)``.

        Parameters
        ----------
        vectors : array_like, shape (3,) or (N, 3)
            Each `vectors[i]` represents a vector in 3D space. A single vector
            can either be specified with shape `(3, )` or `(1, 3)`. The number
            of rotations and number of vectors given must follow standard numpy
            broadcasting rules: either one of them equals unity or they both
            equal each other.
        inverse : boolean, optional
            If True then the inverse of the rotation(s) is applied to the input
            vectors. Default is False.

        Returns
        -------
        rotated_vectors : ndarray, shape (3,) or (N, 3)
            Result of applying rotation on input vectors.
            Shape depends on the following cases:
                - If object contains a single rotation (as opposed to a stack
                  with a single rotation) and a single vector is specified with
                  shape ``(3,)``, then `rotated_vectors` has shape ``(3,)``.
                - In all other cases, `rotated_vectors` has shape ``(N, 3)``,
                  where ``N`` is either the number of rotations or vectors.

        """

        vectors = np.asarray(vectors)
        rot_matrix = np.asarray(rot_matrix)

        if vectors.ndim > 2 or vectors.shape[-1] != 3:
            raise ValueError("Expected input of shape (3,) or (P, 3), "
                             "got {}.".format(vectors.shape))

        single_vector = False
        if vectors.shape == (3,):
            single_vector = True
            vectors = vectors[None, :]

        single_rot = False
        if rot_matrix.shape == (3,3):
            rot_matrix = rot_matrix.reshape((1, 3, 3))
            single_rot = True

        n_vectors = vectors.shape[0]
        n_rotations = len(rot_matrix)

        if n_vectors != 1 and n_rotations != 1 and n_vectors != n_rotations:
            raise ValueError("Expected equal numbers of rotations and vectors "
                             ", or a single rotation, or a single vector, got "
                             "{} rotations and {} vectors.".format(
                             n_rotations, n_vectors))

        if inverse:
            result = np.einsum('ikj,ik->ij', rot_matrix, vectors)
        else:
            result = np.einsum('ijk,ik->ij', rot_matrix, vectors)

        if single_rot and single_vector:
            return result[0]
        else:
            return result

    def group_fofinfo(self) -> Dict[str, np.ndarray]:
        fof_dict = {
                'hubble_param'  : self.hubble_param,
                'comic_time'    : self.comic_time,
                'redshift'      : self.z,
                'OmegaBaryon'   : self.OmegaBaryon,
                'Omega0'        : self.Omega0,
                'OmegaLambda'   : self.OmegaLambda,
                'centre_of_potential' : self.centre_of_potential,
                'r200'          : self.r200,
                'r500'          : self.r500,
                'r2500'         : self.r2500,
                'mfof'          : self.Mtot,
                'm200'          : self.M200,
                'm500'          : self.M500,
                'm2500'         : self.M2500,
                'NumOfSubhalos' : self.NumOfSubhalos
        }
        return fof_dict

    def group_dynamics(self, aperture_radius: float = None) -> Dict[str, np.ndarray]:
        """
        Method that computes the cluster's morphology information from particles within a
        specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the morphology analysis using the relative
        static methods in this Mixin class.
        The method also checks that the necessary datasets are loaded into the cluster object.
        The information is gathered in dictionary-form output and contains the following analysis
        datasets, relative to the given aperture:

            - Inertia tensor (3x3 matrix): np.ndarray
            - Eigenvalues of the inertia tensor, per unit mass 3x np.float
                lambda_i / mass_within_aperture = (semi-axis length)**2
                They represent the semi-axes of the ellipsoid.
            - Eigenvectors of the inertia tensor 3x 1D array
                They are normalised and express the orientation of the semi-axes of the ellipsoid.
            - Triaxiality 1x np.float:
                triaxiality = (a**2-b**2)/(a**2-c**2)
            -Circularity 1x np.float:
                circularity = c/a

        Each of these datasets is structured as follows:

            [
                Dynamics_dataset_allParticleTypes,
                Dynamics_dataset_ParticleType0,
                Dynamics_dataset_ParticleType1,
                Dynamics_dataset_ParticleType4,
            ]

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:2.2f} Mpc.')

        mass = np.zeros(0, dtype=np.float)
        coords = np.zeros((0, 3), dtype=np.float)
        velocity = np.zeros((0, 3), dtype=np.float)
        temperature = np.zeros(0, dtype=np.float)

        aperture_mass = np.zeros(0, dtype=np.float)
        centre_of_mass = np.zeros((0, 3), dtype=np.float)
        zero_momentum_frame = np.zeros((0, 3), dtype=np.float)
        angular_momentum = np.zeros((0, 3), dtype=np.float)
        angular_velocity = np.zeros((0, 3), dtype=np.float)
        specific_angular_momentum = np.zeros(0, dtype=np.float)
        circular_velocity = np.zeros(0, dtype=np.float)
        spin_parameter = np.zeros(0, dtype=np.float)
        substructure_mass = np.zeros(0, dtype=np.float)
        substructure_fraction = np.zeros(0, dtype=np.float)
        thermal_energy = np.zeros(0, dtype=np.float)
        kinetic_energy = np.zeros(0, dtype=np.float)
        dynamical_merging_index = np.zeros(0, dtype=np.float)
        thermodynamic_merging_index = np.zeros(0, dtype=np.float)

        for part_type in ['4', '1', '0']:
            assert hasattr(self, f'partType{part_type}_coordinates')
            assert hasattr(self, f'partType{part_type}_velocity')
            assert hasattr(self, f'partType{part_type}_mass')
            if part_type is '0': assert hasattr(self, f'partType{part_type}_temperature')
            assert hasattr(self, f'partType{part_type}_subgroupnumber')
            radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
            aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
            del radial_dist
            _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
            _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
            _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
            _temperature = getattr(self, f'partType{part_type}_temperature')[aperture_radius_index] if part_type is '0' else np.array([])
            _subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')[aperture_radius_index]
            if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            if part_type is '0' and _temperature.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            if _subgroupnumber.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

            mass = np.concatenate((mass, _mass), axis=0)
            coords = np.concatenate((coords, _coords), axis=0)
            velocity = np.concatenate((velocity, _velocity), axis=0)
            temperature = np.concatenate((temperature, _temperature), axis=0)

            _aperture_mass = np.sum(_mass)
            aperture_mass = np.append(aperture_mass, _aperture_mass)

            _coords_norm = np.subtract(_coords, self.centre_of_potential)
            _centre_of_mass = self.centre_of_mass(_mass, _coords)
            centre_of_mass = np.concatenate((centre_of_mass, _centre_of_mass[None,:]), axis=0)

            _zero_momentum_frame = self.zero_momentum_frame(_mass, _velocity)
            zero_momentum_frame = np.concatenate((zero_momentum_frame, _zero_momentum_frame[None,:]), axis=0)

            _velocity_norm = np.subtract(_velocity, _zero_momentum_frame)
            _angular_momentum = self.angular_momentum(_mass, _coords_norm, _velocity_norm)
            angular_momentum = np.concatenate((angular_momentum, _angular_momentum[None,:]), axis=0)

            _angular_velocity = np.linalg.inv(self.inertia_tensor(_mass, _coords_norm))@_angular_momentum
            angular_velocity = np.concatenate((angular_velocity, _angular_velocity[None,:]), axis=0)

            _specific_angular_momentum = np.linalg.norm(_angular_momentum)/_aperture_mass
            specific_angular_momentum = np.append(specific_angular_momentum, _specific_angular_momentum)

            _circular_velocity = np.sqrt(gravitational_constant*self.mass_units(_aperture_mass)/self.length_units(aperture_radius))/1e3
            circular_velocity = np.append(circular_velocity, _circular_velocity)

            _specific_angular_momentum_SI = self.angular_momentum(
                    self.mass_units(_mass),
                    self.length_units(_coords_norm),
                    self.velocity_units(_velocity_norm)
            ) / np.sum(self.mass_units(_mass))
            _spin_parameter = _specific_angular_momentum_SI/(self.length_units(aperture_radius)*self.velocity_units(_circular_velocity)*np.sqrt(2))
            spin_parameter = np.append(spin_parameter, _spin_parameter)

            sgn_index = np.where(_subgroupnumber == 0)[0]
            _substructure_mass = _aperture_mass - np.sum(_mass[sgn_index])
            substructure_mass = np.append(substructure_mass, _substructure_mass)

            _substructure_fraction = _substructure_mass/_aperture_mass
            substructure_fraction = np.append(substructure_fraction, _substructure_fraction)

            _thermal_energy = self.thermal_energy(self.mass_units(_mass), _temperature)*np.power(10., -46) if part_type is '0' else 0.
            thermal_energy = np.append(thermal_energy, _thermal_energy)

            _kinetic_energy = self.kinetic_energy(self.mass_units(_mass), self.velocity_units(_velocity_norm))*np.power(10., -46)
            kinetic_energy = np.append(kinetic_energy, _kinetic_energy)

            _dynamical_merging_index = np.linalg.norm(self.centre_of_potential-_centre_of_mass)/aperture_radius
            dynamical_merging_index = np.append(dynamical_merging_index, _dynamical_merging_index)

            _thermodynamic_merging_index = _kinetic_energy/_thermal_energy if part_type is '0' else 0.
            thermodynamic_merging_index = np.append(thermodynamic_merging_index, _thermodynamic_merging_index)

            del _mass
            del _velocity
            del _coords
            del _temperature
            del _subgroupnumber
            del _aperture_mass
            del _coords_norm
            del _centre_of_mass
            del _zero_momentum_frame
            del _velocity_norm
            del _angular_momentum
            del _angular_velocity
            del _specific_angular_momentum
            del _circular_velocity
            del _specific_angular_momentum_SI
            del _spin_parameter
            del sgn_index
            del _substructure_mass
            del _substructure_fraction
            del _thermal_energy
            del _kinetic_energy
            del _dynamical_merging_index
            del _thermodynamic_merging_index

        _aperture_mass = np.sum(mass)
        aperture_mass = np.append(aperture_mass, _aperture_mass)

        coords_norm = np.subtract(coords, self.centre_of_potential)
        _centre_of_mass = self.centre_of_mass(mass, coords)
        centre_of_mass = np.concatenate((centre_of_mass, _centre_of_mass[None, :]), axis=0)

        _zero_momentum_frame = self.zero_momentum_frame(mass, velocity)
        zero_momentum_frame = np.concatenate((zero_momentum_frame, _zero_momentum_frame[None, :]), axis=0)

        velocity_norm = np.subtract(velocity, _zero_momentum_frame)
        _angular_momentum = self.angular_momentum(mass, coords_norm, velocity_norm)
        angular_momentum = np.concatenate((angular_momentum, _angular_momentum[None, :]), axis=0)

        _angular_velocity = np.linalg.inv(self.inertia_tensor(mass, coords_norm))@_angular_momentum
        angular_velocity = np.concatenate((angular_velocity, _angular_velocity[None, :]), axis=0)

        _specific_angular_momentum = np.linalg.norm(_angular_momentum) / _aperture_mass
        specific_angular_momentum = np.append(specific_angular_momentum, _specific_angular_momentum)

        _circular_velocity = np.sqrt(gravitational_constant * self.mass_units(_aperture_mass) / self.length_units(aperture_radius)) / 1e3
        circular_velocity = np.append(circular_velocity, _circular_velocity)

        _specific_angular_momentum_SI = self.angular_momentum(
                self.mass_units(mass),
                self.length_units(coords_norm),
                self.velocity_units(velocity_norm)
        )/np.sum(self.mass_units(mass))
        _spin_parameter = _specific_angular_momentum_SI/(self.length_units(aperture_radius)*self.velocity_units(_circular_velocity)*np.sqrt(2))
        spin_parameter = np.append(spin_parameter, _spin_parameter)

        _substructure_mass = np.sum(substructure_mass)
        substructure_mass = np.append(substructure_mass, _substructure_mass)

        _substructure_fraction = _substructure_mass / _aperture_mass
        substructure_fraction = np.append(substructure_fraction, _substructure_fraction)

        _thermal_energy = np.sum(thermal_energy)
        thermal_energy = np.append(thermal_energy, _thermal_energy)

        _kinetic_energy = self.kinetic_energy(self.mass_units(mass), self.velocity_units(velocity_norm))*np.power(10., -46)
        kinetic_energy = np.append(kinetic_energy, _kinetic_energy)

        _dynamical_merging_index = np.linalg.norm(self.centre_of_potential - _centre_of_mass) / aperture_radius
        dynamical_merging_index = np.append(dynamical_merging_index, _dynamical_merging_index)

        _thermodynamic_merging_index = np.sum(thermodynamic_merging_index)
        thermodynamic_merging_index = np.append(thermodynamic_merging_index, _thermodynamic_merging_index)

        del mass
        del coords
        del velocity
        del temperature
        del _aperture_mass
        del coords_norm
        del _centre_of_mass
        del _zero_momentum_frame
        del velocity_norm
        del _angular_momentum
        del _angular_velocity
        del _specific_angular_momentum
        del _circular_velocity
        del _specific_angular_momentum_SI
        del _spin_parameter
        del _substructure_mass
        del _substructure_fraction
        del _thermal_energy
        del _kinetic_energy
        del _dynamical_merging_index
        del _thermodynamic_merging_index

        dynamic_dict = {
                'aperture_mass' : aperture_mass[::-1],
                'centre_of_mass' : centre_of_mass[::-1],
                'zero_momentum_frame' : zero_momentum_frame[::-1],
                'angular_momentum' : angular_momentum[::-1],
                'angular_velocity': angular_velocity[::-1],
                'specific_angular_momentum' : specific_angular_momentum[::-1],
                'circular_velocity' : circular_velocity[::-1],
                'spin_parameter' : spin_parameter[::-1],
                'substructure_mass' : substructure_mass[::-1],
                'substructure_fraction' : substructure_fraction[::-1],
                'thermal_energy' : thermal_energy[::-1],
                'kinetic_energy' : kinetic_energy[::-1],
                'dynamical_merging_index' : dynamical_merging_index[::-1],
                'thermodynamic_merging_index' : thermodynamic_merging_index[::-1]
        }
        return dynamic_dict


    def group_morphology(self, aperture_radius: float = None) -> Dict[str, np.ndarray]:
        """
        Method that computes the cluster's morphology information from particles within a
        specified aperture.
        If the aperture is not specified, it is set by default to the true R500 of the cluster
        radius from the centre of potential and computes the morphology analysis using the relative
        static methods in this Mixin class.
        The method also checks that the necessary datasets are loaded into the cluster object.
        The information is gathered in dictionary-form output and contains the following analysis
        datasets, relative to the given aperture:

            - Inertia tensor (3x3 matrix): np.ndarray
            - Eigenvalues of the inertia tensor, per unit mass 3x np.float
                lambda_i / mass_within_aperture = (semi-axis length)**2
                They represent the semi-axes of the ellipsoid.
            - Eigenvectors of the inertia tensor 3x 1D array
                They are normalised and express the orientation of the semi-axes of the ellipsoid.
            - Triaxiality 1x np.float:
                triaxiality = (a**2-b**2)/(a**2-c**2)
            - Sphericity 1x np.float:
                sphericity = c/a
            - Elongation 1x np.float:
                elongation = c/a

        Each of these datasets is structured as follows:

            [
                Morphology_dataset_allParticleTypes,
                Morphology_dataset_ParticleType0,
                Morphology_dataset_ParticleType1,
                Morphology_dataset_ParticleType4,
            ]

        :param out_allPartTypes: default = False
        :param aperture_radius: default = None (R500)
        :return: expected a numpy array of dimension 1 if all particletypes are combined, or
            dimension 2 if particle types are returned separately.
        """
        if aperture_radius is None:
            aperture_radius = self.r500
            warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

        mass           = np.zeros(0, dtype=np.float)
        coords         = np.zeros((0, 3), dtype=np.float)
        inertia_tensor = np.zeros((0, 9), dtype=np.float)
        eigenvalues    = np.zeros((0, 3), dtype=np.float)
        eigenvectors   = np.zeros((0, 9), dtype=np.float)
        triaxiality    = np.zeros(0, dtype=np.float)
        sphericity     = np.zeros(0, dtype=np.float)
        elongation     = np.zeros(0, dtype=np.float)

        for part_type in ['4', '1', '0']:
            assert hasattr(self, f'partType{part_type}_coordinates')
            assert hasattr(self, f'partType{part_type}_mass')
            radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
            aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
            del radial_dist
            _mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
            _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
            if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
            _coords = np.subtract(_coords, self.centre_of_potential)
            mass = np.concatenate((mass, _mass), axis=0)
            coords = np.concatenate((coords, _coords), axis=0)

            _inertia_tensor = self.inertia_tensor(_mass, _coords)
            inertia_tensor = np.concatenate((inertia_tensor, _inertia_tensor.ravel()[None, :]), axis=0)

            _eigenvalues, _eigenvectors = self.principal_axes_ellipsoid(_inertia_tensor, eigenvalues=True)
            _eigenvalues /= np.sum(_mass)
            # Sort eigenvalues from largest to smallest
            _eigenvalues_sorted  = np.sort(_eigenvalues)[::-1]
            _eigenvectors_sorted = np.zeros_like(_eigenvectors)
            for counter, val in np.ndenumerate(_eigenvalues_sorted):
                index = np.where(_eigenvalues==val)[0]
                _eigenvectors_sorted[counter] = _eigenvectors[index]

            eigenvalues = np.concatenate((eigenvalues, _eigenvalues_sorted[None, :]), axis=0)
            eigenvectors = np.concatenate((eigenvectors, _eigenvectors_sorted.ravel()[None, :]), axis=0)

            _triaxiality = (_eigenvalues_sorted[0]-_eigenvalues_sorted[1])/(_eigenvalues_sorted[0]-_eigenvalues_sorted[2])
            triaxiality = np.append(triaxiality, _triaxiality)

            _sphericity = np.sqrt(_eigenvalues_sorted[2]) / np.sqrt(_eigenvalues_sorted[0])
            sphericity = np.append(sphericity, _sphericity)

            _elongation = np.sqrt(_eigenvalues_sorted[1]) / np.sqrt(_eigenvalues_sorted[0])
            elongation = np.append(elongation, _elongation)

            del _mass
            del _coords
            del _inertia_tensor
            del _eigenvalues
            del _eigenvalues_sorted
            del _eigenvectors_sorted
            del _triaxiality
            del _sphericity
            del _elongation

        _inertia_tensor = self.inertia_tensor(mass, coords)
        inertia_tensor = np.concatenate((inertia_tensor, _inertia_tensor.ravel()[None, :]), axis=0)

        _eigenvalues, _eigenvectors = self.principal_axes_ellipsoid(_inertia_tensor, eigenvalues=True)
        _eigenvalues /= np.sum(mass)
        # Sort eigenvalues from largest to smallest
        _eigenvalues_sorted = np.sort(_eigenvalues)[::-1]
        _eigenvectors_sorted = np.zeros_like(_eigenvectors)
        for counter, val in np.ndenumerate(_eigenvalues_sorted):
            index = np.where(_eigenvalues==val)[0]
            _eigenvectors_sorted[counter] = _eigenvectors[index]
        eigenvalues = np.concatenate((eigenvalues, _eigenvalues_sorted[None, :]), axis=0)
        eigenvectors = np.concatenate((eigenvectors, _eigenvectors_sorted.ravel()[None, :]), axis=0)

        _triaxiality = (_eigenvalues_sorted[0] - _eigenvalues_sorted[1]) / (_eigenvalues_sorted[0] - _eigenvalues_sorted[2])
        triaxiality = np.append(triaxiality, _triaxiality)

        _sphericity = np.sqrt(_eigenvalues_sorted[2])/np.sqrt(_eigenvalues_sorted[0])
        sphericity = np.append(sphericity, _sphericity)

        _elongation = np.sqrt(_eigenvalues_sorted[1])/np.sqrt(_eigenvalues_sorted[0])
        elongation = np.append(elongation, _elongation)

        del mass
        del coords
        del _inertia_tensor
        del _eigenvalues
        del _eigenvalues_sorted
        del _eigenvectors_sorted
        del _triaxiality
        del _sphericity
        del _elongation

        morphology_dict = {
                'inertia_tensor' : inertia_tensor[::-1],
                'eigenvalues'    : eigenvalues[::-1],
                'eigenvectors'   : eigenvectors[::-1],
                'triaxiality'    : triaxiality[::-1],
                'sphericity'     : sphericity[::-1],
                'elongation'     : elongation[::-1],
        }
        return morphology_dict

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

    def comoving_kinetic_energy(self, kinetic_energy):
        """
        Rescales the density from the comoving kinetic_energy to the physical kinetic_energy
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        _kinetic_energy = np.multiply(kinetic_energy, scale_factor)
        _kinetic_energy = np.divide(_kinetic_energy, hubble_par)
        return _kinetic_energy

    def comoving_momentum(self, mom):
        """
        Rescales the momentum from the comoving to the physical
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(mom, np.sqrt(scale_factor) / hubble_par)

    def comoving_ang_momentum(self, angmom):
        """
        Rescales the angular momentum from the comoving to the physical
        """
        hubble_par = self.file_hubble_param()
        redshift = self.file_redshift()
        scale_factor = 1 / (redshift + 1)
        return np.multiply(angmom, np.sqrt(scale_factor**3) / np.power(hubble_par, 2.))


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
    def length_units(len, unit_system='SI'):
        """
		CREATED: 14.02.2019
		LAST MODIFIED: 14.02.2019

		INPUTS: len np.array

				metric system used: 'SI' or 'cgs' or astronomical 'astro'
		"""
        if unit_system == 'SI':
            # m/s
            conv_factor = 1e6 * parsec
        elif unit_system == 'cgs':
            # cm/s
            conv_factor = 1e8 * parsec
        elif unit_system == 'astro':
            # km/s
            conv_factor = 1
        else:
            raise ("[ERROR] Trying to convert length to an unknown metric system.")

        return np.multiply(len, conv_factor)

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
            conv_factor = 1e10 * solar_mass
        elif unit_system == 'cgs':
            # cm/s
            conv_factor = 1e13 * solar_mass
        elif unit_system == 'astro':
            # km/s
            conv_factor = 1e10
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

    @staticmethod
    def energy_units(energy, unit_system='SI'):
        """
		CREATED: 07.03.2019
		LAST MODIFIED: 07.03.2019

		INPUTS: momentum np.array

				metric system used: 'SI' or 'cgs' or astronomical 'astro'
		"""
        if unit_system == 'SI':
            # m/s
            conv_factor = np.power(10., 46)

        else:
            raise ("[ERROR] Trying to convert mass to an unknown metric system.")

        return np.multiply(energy, conv_factor)