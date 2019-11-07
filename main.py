if __name__ == "__main__":
    from clusters_retriever import *
    from cluster_profiler import *
    from mergers import thermal_index, dynamical_index

    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    cluster = Cluster(clusterID = 0, redshift = z_catalogue[-4])
    # print(cluster.group_centre_of_potential())
    # print(cluster.NumOfSubhalos(central_FOF = True))
    # print(cluster.subgroups_number(central_FOF = True))
    print('z = ', cluster.redshift, '\n---------------------')
    print('dynamical_index(cluster), thermal_index(cluster)')
    print(dynamical_index(cluster), '\t\t', thermal_index(cluster))


