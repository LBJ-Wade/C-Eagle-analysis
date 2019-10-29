from os import path
import numpy as np
from matplotlib import pyplot as plt

from clusters_retriever import *
from map_plot_parameters import set_defaults_plot as plotpar

def dist(v, u):
    s = 0
    for v_i, u_i in zip(v, u):
        s += (v_i - u_i)**2
    return s ** 0.5

def dynamical_index(cluster):
    cop = cluster.group_centre_of_potential()
    com = cluster.group_centre_of_mass()
    r500 = cluster.group_r500()
    return dist(com, cop)/r500

def thermal_index(cluster):
    KE = cluster.group_kin_energy()
    TE = cluster.group_therm_energy()
    return TE/KE

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

def mergers_plot():
    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    generate_data = True
    z = 0.101

    for ID in range(0,30):
        if generate_data:
            cluster = Cluster(clusterID = int(ID), redshift = z, subject = 'groups')
            dyn_idx = dynamical_index(cluster)
            therm_idx = thermal_index(cluster)
            print('Process cluster', ID, '\t\t', dyn_idx, '\t\t', therm_idx)
        else:
            # If data are already generated, pick the stored values
            dyn_idx = dynamic_index[ID]
            therm_idx = thermal_index[ID]
        # Draw in matplotlib
        ax.scatter(dyn_idx, therm_idx, color='k')
        ax.annotate(r'${}$'.format(ID), (dyn_idx+0.005, therm_idx+0.005))

    ax.set_xlabel(r'$\mathrm{dynamical~index}$')
    ax.set_ylabel(r'$\mathrm{thermodynamical~index\quad  (Barnes~et~al.,~2017)}$')
    ax.set_title(r'$z = {}$'.format(z))
    ax.set_aspect(1.)
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    plt.show()
    plt.savefig(path.join(ceagle.pathSave, 'Merging_index.png'))
    #print(mrgr_idx)

plotpar()
mergers_plot()

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
