if __name__ == "__main__":
    """
    from clusters_retriever import *
    from cluster_profiler import *
    from mergers import thermal_index, dynamical_index

    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    cluster = Cluster(clusterID = 0, redshift = 0.101)
    # print(cluster.group_centre_of_potential())
    # print(cluster.NumOfSubhalos(central_FOF = True))
    # print(cluster.subgroups_number(central_FOF = True))
    print('z = ', cluster.redshift, '\n---------------------')
    print('dynamical_index(cluster), thermal_index(cluster)')
    # print(dynamical_index(cluster))#, '\t\t', thermal_index(cluster))


    # cop = cluster.group_centre_of_potential()
    # com = cluster.group_centre_of_mass()
    # r500 = cluster.group_r500()
    # print(cop, com, r500)
    """

    import numpy as np
    from matplotlib import pyplot as plt

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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    z = 0.101

    ax.scatter(dyn_index, thermal_index)
    labels = np.array([str(i) for i in range(0, 30)])
    # ax.annotate(labels, (thermal_index , dyn_index))

    ax.set_xlabel(r'$\mathrm{dynamical~index}$')
    ax.set_ylabel(r'$\mathrm{thermodynamical~index}$')
    ax.set_title(r'$z = {}$'.format(z))
    ax.set_aspect(1.)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    plt.show()
    plt.savefig('Merging_index.png')