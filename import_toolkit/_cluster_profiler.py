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
from .memory import free_memory


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

    @staticmethod
    def kinetic_energy(mass, vel):
        ke = 0.5 * mass * np.linalg.norm(vel, axis = 1)**2
        return np.sum(ke)

    @staticmethod
    def thermal_energy(mass, temperature):
        k_B = 1.38064852 * np.power(10, -23.)
        te = 1.5 * k_B * temperature * mass * 0.88 / (1.6735575* np.power(10, -27.))
        return np.sum(te)

    @staticmethod
    def dynamical_merging_index(CoP, CoM, aperture):
        """
        Computed as dynamical_merging_index = || CoM(r) - CoP(r) || / r

        :param CoP: The centre of potential of the cluster
        :param CoM: The centre of mass of the cluster
        :param aperture: The aperture radius (from the CoP) for the particle selection
        :return: float within [0,1]
        """
        displacement = np.linalg.norm(np.subtract(CoM, CoP), axis=1)
        dynamic_index = np.divide(displacement, aperture)
        return dynamic_index

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

    def group_kinetic_energy(self, out_allPartTypes=False, aperture_radius=None):
        """
        Compute the kinetic energy of the particles within a given aperture radius,
        considered from the centre of potential. The function calculates the kinetic energy
        from particle data according to
        kinetic_energy = 1/2 * mass * || velocity ||^2.

        :param out_allPartTypes: expect boolean
            Default = False -> returns the total combined kinetic energy from all particle
            types. If set to True, it returns a numpy.ndarray with the total kinetic energy
            of the 4 individual particle types.

        :param aperture_radius: expect float
            The aperture radius is the distance from the centre of potential of the cluster,
            used to select the particles for the calculation. Everything outside the aperture
            sphere is filtered out.
            The units should be physical Mpc.

        :return: float or np.array(dtype = np.float)
            Returns the kinetic energy of the particles within the aperture sphere.
        """

        Mtot_PartTypes = np.zeros(4, dtype=np.float)

        for idx, part_type in enumerate(['0', '1', '4', '5']):

            # Import data
            mass = self.particle_masses(part_type)
            coords = self.particle_coordinates(part_type)
            group_num = self.group_number_part(part_type)

            # Filter the particles belonging to the
            # GroupNumber FOF == 1, which by definition is centred in the
            # Centre of Potential and is disconnected from other FoF groups.
            radial_dist = np.linalg.norm(np.subtract(coords, self.group_centre_of_potential()), axis=1)

            if aperture_radius is None:
                aperture_radius = self.group_r500()
                print('[ CENTRE OF MASS ]\t==>\tAperture radius set to default R500 true.')

            index = np.where((group_num == self.centralFOF_groupNumber) & (radial_dist < aperture_radius))[0]
            mass = mass[index]
            assert mass.__len__() > 0, "Array is empty - check filtering."
            Mtot_PartTypes[idx] = np.sum(mass)

        if out_allPartTypes:
            return Mtot_PartTypes
        else:
            return np.sum(Mtot_PartTypes)

    def group_mass_aperture(self, out_allPartTypes=False, aperture_radius=None):
        """
        Compute the total mass the particles within a given aperture radius,
        considered from the centre of potential.

        :param out_allPartTypes: expect boolean
            Default = False -> returns the total combined mass from all particle
            types. If set to True, it returns a numpy.ndarray with the total kinetic energy
            of the 4 individual particle types.

        :param aperture_radius: expect float
            The aperture radius is the distance from the centre of potential of the cluster,
            used to select the particles for the calculation. Everything outside the aperture
            sphere is filtered out.
            The units should be physical Mpc.

        :return: float or np.array(dtype = np.float)
            Returns the kinetic energy of the particles within the aperture sphere.
        """

        Mtot_PartTypes = np.zeros(4, dtype=np.float)

        for idx, part_type in enumerate(['0', '1', '4', '5']):

            # Import data
            mass = self.particle_masses(part_type)
            coords = self.particle_coordinates(part_type)
            group_num = self.group_number_part(part_type)

            # Filter the particles belonging to the
            # GroupNumber FOF == 1, which by definition is centred in the
            # Centre of Potential and is disconnected from other FoF groups.
            radial_dist = np.linalg.norm(np.subtract(coords, self.group_centre_of_potential()), axis=1)

            if aperture_radius is None:
                aperture_radius = self.group_r500()
                print('[ CENTRE OF MASS ]\t==>\tAperture radius set to default R500 true.')

            index = np.where((group_num == self.centralFOF_groupNumber) & (radial_dist < aperture_radius))[0]
            mass = mass[index]
            assert mass.__len__() > 0, "Array is empty - check filtering."
            Mtot_PartTypes[idx] = np.sum(mass)

        if out_allPartTypes:
            return Mtot_PartTypes
        else:
            return np.sum(Mtot_PartTypes)

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
            print('[ CENTRE OF MASS ]\t==>\tAperture radius set to default R500 true.')

        if out_allPartTypes:

            CoM_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass   = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                assert _mass.__len__() > 0,   "Array is empty - check filtering."
                assert _coords.__len__() > 0, "Array is empty - check filtering."

                centre_of_mass = self.centre_of_mass(_mass, _coords)
                CoM_PartTypes = np.append(CoM_PartTypes, [centre_of_mass], axis=0)

            return CoM_PartTypes

        else:

            mass   = np.zeros(0, dtype=np.float)
            coords = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass   = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                assert _mass.__len__() > 0,   "Array is empty - check filtering."
                assert _coords.__len__() > 0, "Array is empty - check filtering."
                mass   = np.append(mass, [_mass], axis=0)
                coords = np.append(coords, [_coords], axis=0)

            return self.centre_of_mass(mass, coords)

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
            print('[ ZERO MOMENTUM FRAME ]\t==>\tAperture radius set to default R500 true.')

        if out_allPartTypes:

            ZMF_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                assert _mass.__len__() > 0,     "Array is empty - check filtering."
                assert _velocity.__len__() > 0, "Array is empty - check filtering."

                zmf = self.zero_momentum_frame(_mass, _velocity)
                ZMF_PartTypes = np.append(ZMF_PartTypes, [zmf], axis=0)

            return ZMF_PartTypes

        else:

            mass     = np.zeros(0, dtype=np.float)
            velocity = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                assert _mass.__len__() > 0,     "Array is empty - check filtering."
                assert _velocity.__len__() > 0, "Array is empty - check filtering."
                mass     = np.append(mass, [_mass], axis=0)
                velocity = np.append(velocity, [_velocity], axis=0)

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
            print('[ ZERO MOMENTUM FRAME ]\t==>\tAperture radius set to default R500 true.')

        if out_allPartTypes:

            ZMF_PartTypes = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords   = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                assert _mass.__len__() > 0,     "Array is empty - check filtering."
                assert _coords.__len__() > 0,   "Array is empty - check filtering."
                assert _velocity.__len__() > 0, "Array is empty - check filtering."

                # Rescale coordinates and velocity
                _coords   = np.subtract(_coords, self.centre_of_potential)
                _velocity = np.subtract(_velocity, self.group_zero_momentum_frame(aperture_radius=aperture_radius))
                zmf = self.angular_momentum(_mass, _coords, _velocity)
                ZMF_PartTypes = np.append(ZMF_PartTypes, [zmf], axis=0)

            return ZMF_PartTypes

        else:

            mass     = np.zeros(0, dtype=np.float)
            coords   = np.zeros((0, 3), dtype=np.float)
            velocity = np.zeros((0, 3), dtype=np.float)

            for part_type in ['0', '1', '4', '5']:
                assert hasattr(self, f'partType{part_type}_coordinates')
                assert hasattr(self, f'partType{part_type}_velocity')
                assert hasattr(self, f'partType{part_type}_mass')
                radial_dist = np.linalg.norm(np.subtract(getattr(self, f'partType{part_type}_coordinates'),
                                                         self.centre_of_potential), axis=1)

                aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
                free_memory(['radial_dist'])
                _mass     = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
                _coords   = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
                _velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
                assert _mass.__len__() > 0,     "Array is empty - check filtering."
                assert _coords.__len__() > 0,   "Array is empty - check filtering."
                assert _velocity.__len__() > 0, "Array is empty - check filtering."
                mass     = np.append(mass, [_mass], axis=0)
                coords   = np.append(coords, [_coords], axis=0)
                velocity = np.append(velocity, [_velocity], axis=0)

            # Rescale coordinates and velocity
            coords   = np.subtract(coords, self.centre_of_potential)
            velocity = np.subtract(velocity, self.group_zero_momentum_frame(aperture_radius=aperture_radius))
            return self.angular_momentum(mass, coords, velocity)



    def generate_apertures(self):
        """
        Generate an array of apertures for calculating global properties of the clusters.
        The apertures use both R2500 and R200 units:

        :return: (np.ndarray)
            The array with 100 different apertures, ranging from 0.5 R2500 to 5*R200
            NOTE: the apertures are returned in the PHYSICAL frame.
        """
        if self.r2500 > 0. and self.r200 > 0.:
            apertures = np.logspace(np.log10(0.5 * self.r2500), np.log10(5 * self.r200), 20)
        else:
            apertures = -1
            print(ValueError)
            print('Issue encountered at ', self.clusterID, self.redshift)

        return apertures

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