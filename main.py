if __name__ == "__main__":

    from clusters_retriever import *
    from cluster_profiler import *

    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    cluster = Cluster(clusterID = 0, redshift = 0.101)
    # print(cluster.group_centre_of_potential())
    # print(cluster.NumOfSubhalos(central_FOF = True))
    # print(cluster.subgroups_number(central_FOF = True))
    print('z = ', cluster.redshift, '\n---------------------')
    print('dynamical_index(cluster), thermal_index(cluster)')
    # print(dynamical_index(cluster))#, '\t\t', thermal_index(cluster))


    cop = cluster.group_centre_of_potential()
    com = cluster.group_centre_of_mass()
    r500 = cluster.group_r500()
    print(cop, com, r500)
