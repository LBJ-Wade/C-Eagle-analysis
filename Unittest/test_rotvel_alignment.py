import os
import sys
import unittest
import numpy as np
import h5py as h5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster

np.set_printoptions(suppress=True)

class TestRotVel(unittest.TestCase):

    def test_rotvel_alignment_GAS(self):

        # Read in celr_e | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/groups_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_tab_029_z000p000.0.hdf5'), 'r') as f:

            hd5set = f['/FOF/Group_R_Crit500']
            r500 = hd5set[...][0]

            hd5set = f['/FOF/GroupCentreOfPotential']
            CoP = hd5set[...][0]

        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/particledata_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_029_z000p000.0.hdf5'), 'r') as f:

            hd5set = f['/PartType0/GroupNumber']
            group_number = hd5set[...]

            hd5set = f['/PartType0/Coordinates']
            coordinates = hd5set[...]

            hd5set = f['/PartType0/Velocity']
            velocity = hd5set[...]

            hd5set = f['/PartType0/Mass']
            mass = hd5set[...]

            hd5set = f['/PartType0/Temperature']
            temperature = hd5set[...]

        # Convert in cluster-centric radial coordinates
        coordinates_radial = np.sqrt(
            (coordinates[:, 0] - CoP[0]) ** 2 +
            (coordinates[:, 1] - CoP[1]) ** 2 +
            (coordinates[:, 2] - CoP[2]) ** 2
        )

        # Rescale coordinates to the CoP
        coordinates = np.asarray([
            (coordinates[:, 0] - CoP[0]),
            (coordinates[:, 1] - CoP[1]),
            (coordinates[:, 2] - CoP[2])
        ]).T

        # Apply masks:
        #   aperture radius = R500_true
        #   no equation of state
        #   central FOF group
        mask = np.where(
            (coordinates_radial < r500) &
            (temperature > 10**5) &
            (group_number == 1)
        )[0]

        coordinates = coordinates[mask]
        velocity = velocity[mask]
        mass = mass[mask]

        # Compute peculiar velocity
        pec_velocity = np.sum(velocity*mass[:, None], axis = 0)/np.sum(mass)

        velocity_ZMF = np.asarray([
            (velocity[:, 0] - pec_velocity[0]),
            (velocity[:, 1] - pec_velocity[1]),
            (velocity[:, 2] - pec_velocity[2])
        ]).T

        # Compute angular momentum as r [cross] v
        linear_momentum_r = velocity*mass[:, None]
        ang_momentum = np.sum(np.cross(coordinates, linear_momentum_r), axis = 0)/np.sum(mass)

        # Compute angle between pec_velocity and ang_momentum
        delta_theta = np.arccos(np.dot(pec_velocity, ang_momentum) / (np.linalg.norm(pec_velocity) * np.linalg.norm(
            ang_momentum))) * 180/np.pi

        # Display results
        print(f"{' celr_e | halo 0 | z=0 ':-^60}\nComoving frame: {True}\nUnits: {'Gadget units'}\nParticles: {'Gas'}\n")
        # print(f"Shape of group_number: {group_number.shape}")
        # print(f"Shape of coordinates: {coordinates.shape}")
        # print(f"Shape of velocity: {velocity.shape}")
        # print(f"Shape of mass: {mass.shape}")
        # print(f"Shape of temperature: {temperature.shape}")
        # print(f"Shape of mask: {mask.shape}")
        # print(f"Shape of velocity_ZMF: {velocity_ZMF.shape}")
        # print(f"Shape of pec_velocity: {pec_velocity.shape}")
        # print(f"Shape of ang_momentum: {ang_momentum.shape}")
        # print('\n')
        print(f"pec_velocity = {pec_velocity}")
        print(f"ang_momentum = {ang_momentum}")
        print(f"delta_theta = {delta_theta}")

    def test_rotvel_alignment_DM(self):
        # Read in celr_e | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/groups_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_tab_029_z000p000.0.hdf5'), 'r') as f:
            hd5set = f['/FOF/Group_R_Crit500']
            r500 = hd5set[...][0]

            hd5set = f['/FOF/GroupCentreOfPotential']
            CoP = hd5set[...][0]

        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/particledata_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_029_z000p000.0.hdf5'), 'r') as f:
            hd5set = f['/PartType1/GroupNumber']
            group_number = hd5set[...]

            hd5set = f['/PartType1/Coordinates']
            coordinates = hd5set[...]

            hd5set = f['/PartType1/Velocity']
            velocity = hd5set[...]

            h5dset = f["/Header"]
            DM_particleMass   = h5dset.attrs.get('MassTable', default=None)[1]
            DM_Numberparticle = h5dset.attrs.get('NumPart_Total', default=None)[1]
            mass = np.ones(DM_Numberparticle) * DM_particleMass


        # Convert in cluster-centric radial coordinates
        coordinates_radial = np.sqrt(
            (coordinates[:, 0] - CoP[0]) ** 2 +
            (coordinates[:, 1] - CoP[1]) ** 2 +
            (coordinates[:, 2] - CoP[2]) ** 2
        )

        # Rescale coordinates to the CoP
        coordinates = np.asarray([
            (coordinates[:, 0] - CoP[0]),
            (coordinates[:, 1] - CoP[1]),
            (coordinates[:, 2] - CoP[2])
        ]).T

        # Apply masks:
        #   aperture radius = R500_true
        #   no equation of state
        #   central FOF group
        mask = np.where(
            (coordinates_radial < r500) &
            (group_number == 1)
        )[0]

        coordinates = coordinates[mask]
        velocity = velocity[mask]
        mass = mass[mask]

        # Compute peculiar velocity
        pec_velocity = np.sum(velocity * mass[:, None], axis=0) / np.sum(mass)

        velocity_ZMF = np.asarray([
            (velocity[:, 0] - pec_velocity[0]),
            (velocity[:, 1] - pec_velocity[1]),
            (velocity[:, 2] - pec_velocity[2])
        ]).T

        # Compute angular momentum as r [cross] v
        linear_momentum_r = velocity_ZMF * mass[:, None]
        ang_momentum = np.sum(np.cross(coordinates, linear_momentum_r), axis=0) / np.sum(mass)

        # Compute angle between pec_velocity and ang_momentum
        delta_theta = np.arccos(np.dot(pec_velocity, ang_momentum) / (np.linalg.norm(pec_velocity) * np.linalg.norm(
            ang_momentum))) * 180 / np.pi

        # Display results
        print(f"{' celr_e | halo 0 | z=0 ':-^60}\nComoving frame: {True}\nUnits: {'Gadget units'}\nParticles: {'DM'}\n")
        # print(f"Shape of group_number: {group_number.shape}")
        # print(f"Shape of coordinates: {coordinates.shape}")
        # print(f"Shape of velocity: {velocity.shape}")
        # print(f"Shape of mass: {mass.shape}")
        # print(f"Shape of mask: {mask.shape}")
        # print(f"Shape of velocity_ZMF: {velocity_ZMF.shape}")
        # print(f"Shape of pec_velocity: {pec_velocity.shape}")
        # print(f"Shape of ang_momentum: {ang_momentum.shape}")
        # print('\n')
        print(f"pec_velocity = {pec_velocity}")
        print(f"ang_momentum = {ang_momentum}")
        print(f"delta_theta = {delta_theta}")

    def test_software_rotvel_alignment_GAS(self):
        print(f"{' SOFTWARE TEST ':*^60}")
        data_required = {'partType0': ['mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
                         'partType1': ['mass', 'coordinates', 'velocity'],
                         'partType4': ['mass', 'coordinates', 'velocity'],
                         'partType5': ['mass', 'coordinates', 'velocity']}

        for sim in ['celr_e', 'celr_b', 'macsis']:

            cluster = Cluster(simulation_name=sim,
                              clusterID=0,
                              redshift='z000p000',
                              comovingframe=False,
                              requires=data_required)

            cluster.info()

            print(f"\n {sim}{' | halo 0 | z=0 ':-^60}")
            pec_velocity = cluster.group_zero_momentum_frame(aperture_radius=cluster.r200)
            ang_momentum = cluster.group_angular_momentum(aperture_radius=cluster.r200)
            print(f"pec_velocity = {pec_velocity}")
            print(f"ang_momentum = {ang_momentum}")
            print("cluster.angle_between_vectors(pec_velocity, ang_momentum)",
                  cluster.angle_between_vectors(pec_velocity, ang_momentum))


if __name__ == '__main__':
    unittest.main()


