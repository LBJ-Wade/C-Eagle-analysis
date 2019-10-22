from matplotlib import pyplot as plt
import numpy as np

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
    r200 = cluster.group_r200()
    return dist(com, cop)/r200

def thermal_index():
    """ From Barnes et al., 2017 """
    return [0.11, 0.24, 0.05, 0.06, 0.11, 0.09, 0.09, 0.09, 0.09,
            0.16, 0.05, 0.21, 0.08, 0.06, 0.31, 0.22, 0.12, 0.25,
            0.09, 0.30, 0.12, 0.29, 0.12, 0.26, 0.13, 0.31, 0.08,
            0.24, 0.13, 0.30]

def dynamic_index():
    """ Computed by Altamura, Oct 2019 """
    return  [0.07550353, 0.11060472, 0.02386166, 0.05197934, 0.12173181, 0.14476155,
             0.1495804,  0.11658911, 0.1523977,  0.17623171, 0.10816798, 0.07357231,
             0.02656429, 0.01441143, 0.10300025, 0.09864411, 0.19084678, 0.10902855,
             0.05147959, 0.12121805, 0.0963832,  0.22812534, 0.09900531, 0.26436435,
             0.07748996, 0.20402204, 0.01852897, 0.07137036, 0.0792575,  0.10791006]

def mergers_plot():
    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    mrgr_idx = np.array([], dtype = np.float)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    generate_data = False

    for ID in range(0,30):
        if generate_data:
            cluster = Cluster(clusterID = int(ID), redshift = z_catalogue[-1], subject = 'groups')
            dyn_idx = dynamical_index(cluster)
            print('Process cluster', ID, '\t\t', dyn_idx)
            mrgr_idx = np.concatenate([mrgr_idx, [dyn_idx]])
        else:
            # If data are already generated, pick the stored values
            dyn_idx = dynamic_index()[ID]
        # Draw in matplotlib
        ax.scatter(dyn_idx, thermal_index()[ID], color='k')
        ax.annotate(r'${}$'.format(ID), (dyn_idx+0.005, thermal_index()[ID]+0.005))

    ax.set_xlabel(r'$\mathrm{dynamical~index}$')
    ax.set_ylabel(r'$\mathrm{thermodynamical~index\quad  (Barnes~et~al.,~2017)}$')
    ax.set_aspect(1.)
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0.5*np.min(dynamic_index()), 1.2*np.max(dynamic_index())])
    ax.set_ylim([0.5*np.min(thermal_index()), 1.2*np.max(thermal_index())])
    # plt.show()
    plt.savefig('Merging index.png')
    #print(mrgr_idx)

plotpar()
mergers_plot()

"""
NOTES
OUTPUT
--------------------------------------------------------------------------
Process cluster 0 		 0.07550353083679252
Process cluster 1 		 0.11060471541374275
Process cluster 2 		 0.023861661562625106
Process cluster 3 		 0.051979339898497255
Process cluster 4 		 0.1217318058492607
Process cluster 5 		 0.14476155071078126
Process cluster 6 		 0.14958040423608657
Process cluster 7 		 0.11658910869418349
Process cluster 8 		 0.15239770234106303
Process cluster 9 		 0.17623171015324904
Process cluster 10 		 0.10816798105327913
Process cluster 11 		 0.07357231320545506
Process cluster 12 		 0.02656429238448933
Process cluster 13 		 0.014411425930184962
Process cluster 14 		 0.1030002493409389
Process cluster 15 		 0.09864410664181773
Process cluster 16 		 0.19084677551005677
Process cluster 17 		 0.10902855395249048
Process cluster 18 		 0.05147959162247487
Process cluster 19 		 0.12121804538480342
Process cluster 20 		 0.09638320274296963
Process cluster 21 		 0.22812534327223366
Process cluster 22 		 0.09900530793998853
Process cluster 23 		 0.26436435097860267
Process cluster 24 		 0.07748995997262832
Process cluster 25 		 0.2040220393769189
Process cluster 26 		 0.01852897205538349
Process cluster 27 		 0.0713703632782728
Process cluster 28 		 0.07925750369294346
Process cluster 29 		 0.10791005759051799
[0.07550353 0.11060472 0.02386166 0.05197934 0.12173181 0.14476155
 0.1495804  0.11658911 0.1523977  0.17623171 0.10816798 0.07357231
 0.02656429 0.01441143 0.10300025 0.09864411 0.19084678 0.10902855
 0.05147959 0.12121805 0.0963832  0.22812534 0.09900531 0.26436435
 0.07748996 0.20402204 0.01852897 0.07137036 0.0792575  0.10791006]
"""
