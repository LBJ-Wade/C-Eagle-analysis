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
from memory import free_memory


class Mixin:


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
    def angular_momentum(mass, velocity, position):
        """Defined as L = m(r CROSS v)"""
        rxv = np.cross(position, velocity)
        assert (type(rxv) == np.ndarray) and (type(mass) == np.ndarray)
        ang_mom = rxv * mass[:, None]
        return np.sum(ang_mom, axis = 0), np.sum(mass)

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


    def group_centre_of_mass(self, out_allPartTypes=False, aperture_radius = None):
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

            # Filter the particles belonging to the
            # GroupNumber FOF == 1, which by definition is centred in the
            # Centre of Potential and is disconnected from other FoF groups.
            radial_dist = np.linalg.norm(np.subtract(coords, self.group_centre_of_potential()), axis = 1)

            if aperture_radius is None:
                aperture_radius = self.group_r500()
                print('[ CENTRE OF MASS ]\t==>\tAperture radius set to default R500 true.')

            # print('centralFOF_groupNumber:', self.centralFOF_groupNumber)
            index = np.where((group_num == self.centralFOF_groupNumber) & (radial_dist < aperture_radius))[0]
            mass = mass[index]
            coords = coords[index]
            assert mass.__len__() > 0, "Array is empty - check filtering."
            assert coords.__len__() > 0, "Array is empty - check filtering."
            # print('Computing CoM ==> PartType {0} ok! {1} particles selected.'.format(part_type, mass.__len__()))

            # Compute CoM for each particle type
            centre_of_mass, sum_of_masses = self.centre_of_mass(mass, coords)
            CoM_PartTypes = np.append(CoM_PartTypes, [centre_of_mass], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

        if out_allPartTypes:
            return CoM_PartTypes, Mtot_PartTypes
        else:
            return self.centre_of_mass(Mtot_PartTypes, CoM_PartTypes)

    def group_zero_momentum_frame(self, out_allPartTypes=False, aperture_radius = None):
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

            # Filter the particles belonging to the
            # GroupNumber FOF == 1, which by definition is centred in the
            # Centre of Potential and is disconnected from other FoF groups.
            radial_dist = np.linalg.norm(np.subtract(coords, self.group_centre_of_potential()), axis=1)

            if aperture_radius is None:
                aperture_radius = self.group_r500()
                print('[ ZERO MOMENTUM ]\t==>\tAperture radius set to default R500 true.')

            index = np.where((group_num == self.centralFOF_groupNumber) & (radial_dist < aperture_radius))[0]
            mass = mass[index]
            vel = vel[index]
            assert mass.__len__() > 0, "Array is empty - check filtering.."
            assert vel.__len__() > 0, "Array is empty - check filtering."
            # print('Computing ZMF ==> PartType {} ok!'.format(part_type))

            # Compute *local* ZMF for each particle type
            zero_momentum, sum_of_masses = self.zero_momentum_frame(mass, vel)
            ZMF_PartTypes = np.append(ZMF_PartTypes, [zero_momentum], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

        if out_allPartTypes:
            return ZMF_PartTypes, Mtot_PartTypes
        else:
            return self.zero_momentum_frame(Mtot_PartTypes, ZMF_PartTypes)

    def group_angular_momentum(self, out_allPartTypes=False, aperture_radius = None):
        """
        out_allPartTypes = (bool)
            if True outputs the zero_momentum_frame and sum of masses of each
            partType separately in arrays

            if False outputs the overall zero_momentum_frame and sum of masses
            of the whole cluster.

        Returns the zero_momentum_frame of the cluster for a ALL particle types,
        except for lowres_DM (2, 3).
        """
        angular_momentum_PartTypes = np.zeros((0, 3), dtype=np.float)
        Mtot_PartTypes = np.zeros(0, dtype=np.float)

        CoP_coords = self.group_centre_of_potential()

        for part_type in ['0', '1', '4', '5']:
            # Import data
            mass = self.particle_masses(part_type)
            vel = self.particle_velocity(part_type)
            coords = self.particle_coordinates(part_type)
            group_num = self.group_number_part(part_type)


            # Filter the particles belonging to the
            # GroupNumber FOF == 1, which by definition is centred in the
            # Centre of Mass and is disconnected from other FoF groups.
            # NOTE: the CoM is only present here since the rotation of the
            # cluster occurs about the CoM.
            radial_dist = np.linalg.norm(np.subtract(coords, CoP_coords), axis=1)

            if aperture_radius is None:
                aperture_radius = self.group_r500()
                print('[ ANG MOMENTUM ]\t==>\tAperture radius set to default R500 true.')

            index = np.where((group_num == self.centralFOF_groupNumber) & (radial_dist < aperture_radius))[0]
            mass = mass[index]
            coords = coords[index]
            vel = vel[index]
            assert mass.__len__() > 0, "Array is empty - check filtering.."
            assert vel.__len__() > 0, "Array is empty - check filtering."
            assert coords.__len__() > 0, "Array is empty - check filtering."
            # print('Computing angular_momentum ==> PartType {} ok!'.format(part_type))

            # Compute *local* angular momentum for each particle type
            zero_ang_momentum, sum_of_masses = self.angular_momentum(mass, vel, coords)
            angular_momentum_PartTypes = np.append(angular_momentum_PartTypes, [zero_ang_momentum], axis=0)
            Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

        if out_allPartTypes:
            return angular_momentum_PartTypes, Mtot_PartTypes
        else:
            return np.sum(angular_momentum_PartTypes, axis = 0), np.sum(Mtot_PartTypes)


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
        return rotation_matrix

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