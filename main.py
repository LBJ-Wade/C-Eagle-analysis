if __name__ == "__main__":
    from clusters_retriever import *
    from cluster_profiler import *
    from mergers import thermal_index as th

    ceagle = Simulation()
    z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    cluster = Cluster(clusterID = 0, redshift = z_catalogue[-1])
    print(cluster.group_centre_of_potential())
    print(cluster.NumOfSubhalos(central_FOF = True))
    # print(cluster.subgroups_number(central_FOF = True))
    print(th(cluster))


