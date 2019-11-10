import os
from os import path
from matplotlib import pyplot as plt
import h5py
from scipy.spatial import distance
from clusters_retriever import *
import cluster_profiler as profile
import map_plot_parameters as plotpar
# import cluster_profiler as profile
"""
GLOBAL VARS
"""
ceagle = Simulation()
# z_catalogue = ceagle.get_redshiftAllowed(dtype=float)
__pathSave__ = ceagle.pathSave + '/merger_index/'
k_B = 1.38064852e-23


def dynamical_index(cluster):
    cop = cluster.group_centre_of_potential()
    com = cluster.group_centre_of_mass()
    r500 = cluster.group_r500()
    return distance.euclidean(cop, com)/r500

def thermal_index(cluster):
    plot_groups = 'FoF'
    part_type = '0'

    mass = cluster.particle_masses(part_type)
    coordinates = cluster.particle_coordinates(part_type)
    velocities = cluster.particle_velocity(part_type)
    group_number = cluster.group_number_part(part_type)
    subgroup_number = cluster.subgroup_number_part(part_type)
    temperatures = cluster.particle_temperature(part_type)
    tot_rest_frame = cluster.cluster_average_velocity(velocities)



    h = cluster.file_hubble_param()
    redshift = cluster.redshift
    r500 = cluster.group_r500()

    # Retrieve coordinates & velocities
    group_CoP = cluster.group_centre_of_potential()
    x = coordinates[:, 0] - group_CoP[0]
    y = coordinates[:, 1] - group_CoP[1]
    z = coordinates[:, 2] - group_CoP[2]
    vx = velocities[:, 0] - tot_rest_frame[0]
    vy = velocities[:, 1] - tot_rest_frame[1]
    vz = velocities[:, 2] - tot_rest_frame[2]

    # Rescale to comoving coordinates
    x = profile.comoving_length(x, h, redshift)
    y = profile.comoving_length(y, h, redshift)
    z = profile.comoving_length(z, h, redshift)
    r500 = profile.comoving_length(r500, h, redshift)
    vx = profile.comoving_velocity(vx, h, redshift)
    vy = profile.comoving_velocity(vy, h, redshift)
    vz = profile.comoving_velocity(vz, h, redshift)
    vx = profile.velocity_units(vx, unit_system='SI')
    vy = profile.velocity_units(vy, unit_system='SI')
    vz = profile.velocity_units(vz, unit_system='SI')
    mass = profile.comoving_mass(mass, h, redshift)
    mass = profile.mass_units(mass, unit_system='SI')

    # Compute radial distance
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Select particles within 5*r200
    if plot_groups == 'FoF':
        index = np.where((r < 5 * r500) & (group_number > -1) & (subgroup_number < 5000) & (subgroup_number > -1))[0]
    elif plot_groups == 'subgroups':
        index = np.where((r < 5 * r500) & (group_number > -1) & (subgroup_number > 0))[0]
    else:
        print("[ERROR] The (sub)groups you are trying to plot are not defined.")
        exit(1)

    free_memory(['x', 'y', 'z'], invert=False)
    mass = mass[index]
    temperatures = temperatures[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    KE = 0.5 * mass * (vx**2 + vy**2 + vz**2)
    TE = 1.5 * k_B * temperatures * mass * 0.88 / (1.6735575e-27)
    thermdyn = np.sum(KE)/np.sum(TE)
    free_memory([thermdyn], invert=True)
    return thermdyn

def gen_data():
    """
    Generates the dynamical index and thermodynamical index
    for different C-EAGLE clusters.
    :return:
    Dictionary containing two arrays of floats.
    """

    z = 0.101
    merg_indices = {'dynamical_index':      [],
                    'thermodynamic_index':  []}
    for ID in range(0, 1):
        cluster = Cluster(clusterID=int(ID), redshift=z, subject='groups')
        dyn_idx = dynamical_index(cluster)
        therm_idx = thermal_index(cluster)
        print('Process cluster', ID, '\t\tdyn: ', dyn_idx, '\t\ttherm: ', therm_idx)
        merg_indices['dynamical_index'].append(dyn_idx)
        merg_indices['thermodynamic_index'].append(therm_idx)

    # Create a trigger that allows the save operation to proceed
    trigger = False
    if (len(merg_indices['dynamical_index']) == ceagle.totalClusters and
        len(merg_indices['thermodynamic_index']) == ceagle.totalClusters and
        np.all(np.isfinite(merg_indices['dynamical_index'])) == True and
        np.all(np.isfinite(merg_indices['thermodynamic_index'])) == True):

        trigger = True

    else:
        raise('Generated data have something wrong.')

    return merg_indices, trigger

def save_data(*out_data):
    """
    Saves intermediate data into an hdf5 file.
    SaveDirectory: /cosma6/data/...
    :return:
    None - saves the hdf5 file.
    """
    merg_indices = out_data[0]
    trigger      = out_data[1]
    if trigger:
        hf = h5py.File(__pathSave__ + 'merg_indices.h5', 'w')
        hf.create_dataset('dynamical_index', data=merg_indices['dynamical_index'])
        hf.create_dataset('thermodynamic_index', data=merg_indices['thermodynamic_index'])
        hf.close()

def read_data(file):
    if path.isfile(__pathSave__+file) and os.access(__pathSave__+file, os.R_OK):
        hf = h5py.File(__pathSave__ + 'merg_indices.h5', 'r')
        dyn = np.array(hf.get('dynamical_index'))
        ther = np.array(hf.get('thermodynamic_index'))
        hf.close()

        return {'dynamical_index':      dyn,
                'thermodynamic_index':  ther}

    else:
        print('[Warning]\thdf5 intermediate data cannot be found.')
        gen_trigger = input('Would you like to generate a new set of data? ([y]/n)  ')
        if gen_trigger == 'y':
            save_data(gen_data())
        elif gen_trigger == 'n':
            quit()


def plot_data(data):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    z = 0.101

    ax.scatter(data['dynamical_index'], data['thermodynamic_index'], color='k')

    labels = np.array([str(i) for i in range(0, 30)])
    ax.annotate(labels, (data['dynamical_index']+0.005, data['thermodynamic_index']+0.005))

    ax.set_xlabel(r'$\mathrm{dynamical~index}$')
    ax.set_ylabel(r'$\mathrm{thermodynamical~index}$')
    ax.set_title(r'$z = {}$'.format(z))
    ax.set_aspect(1.)
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    plt.show()
    plt.savefig(path.join(__pathSave__, 'Merging_index.png'))
    #print(mrgr_idx)



if __name__ == "__main__":
    merg_indices = read_data('merg_indices.hdf5')
    plot_data(merg_indices)




"""
thermal_index = [0.11, 0.24, 0.05, 0.06, 0.11, 0.09, 0.09, 0.09, 0.09,
                0.16, 0.05, 0.21, 0.08, 0.06, 0.31, 0.22, 0.12, 0.25,
                0.09, 0.30, 0.12, 0.29, 0.12, 0.26, 0.13, 0.31, 0.08,
                0.24, 0.13, 0.30]

dyn_index = [
0.09471844251025154,
0.05266533206336863,
0.09905338542130626,
0.07798685947928644,
0.35989168869256366,
0.1594701268260165,
0.17802302052289637,
0.16221126160351654,
0.31054939530620845,
0.4543237838050222,
0.1863821757707459,
0.11039822777010391,
0.20690930447464262,
0.1812836797295026,
0.11718308096238546,
0.26205341914374924,
0.5667122466696966,
0.09638546084697633,
0.11219876490148235,
0.0896929411187384,
0.1589376578507554,
0.31729299224711743,
0.14065290705733102,
0.8370545989148982,
0.24274851651932408,
0.1101912094773132,
0.04310107750859479,
0.6132082764750534,
0.11882789532867002,
0.18064748591247506]
"""



"""
NOTES
OUTPUT
--------------------------------------------------------------------------
Process cluster 0 		 0.09471844251025154
Process cluster 1 		 0.05266533206336863
Process cluster 2 		 0.09905338542130626
Process cluster 3 		 0.07798685947928644
Process cluster 4 		 0.35989168869256366
Process cluster 5 		 0.1594701268260165
Process cluster 6 		 0.17802302052289637
Process cluster 7 		 0.16221126160351654
Process cluster 8 		 0.31054939530620845
Process cluster 9 		 0.4543237838050222
Process cluster 10 		 0.1863821757707459
Process cluster 11 		 0.11039822777010391
Process cluster 12 		 0.20690930447464262
Process cluster 13 		 0.1812836797295026
Process cluster 14 		 0.11718308096238546
Process cluster 15 		 0.26205341914374924
Process cluster 16 		 0.5667122466696966
Process cluster 17 		 0.09638546084697633
Process cluster 18 		 0.11219876490148235
Process cluster 19 		 0.0896929411187384
Process cluster 20 		 0.1589376578507554
Process cluster 21 		 0.31729299224711743
Process cluster 22 		 0.14065290705733102
Process cluster 23 		 0.8370545989148982
Process cluster 24 		 0.24274851651932408
Process cluster 25 		 0.1101912094773132
Process cluster 26 		 0.04310107750859479
Process cluster 27 		 0.6132082764750534
Process cluster 28 		 0.11882789532867002
Process cluster 29 		 0.18064748591247506

"""
