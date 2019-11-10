"""
main.py

This file contains the main() function and is used for testing
purposes. It can also be linked to shell arguments for profiling.

"""

__PROFILE__ = True

def main():

    from clusters_retriever import *
    from cluster_profiler import *

    # ceagle = Simulation()
    # z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
    cluster = Cluster(clusterID = 0, redshift = 0.101)


    cop = cluster.group_centre_of_potential()
    com = cluster.group_centre_of_mass()
    r500 = cluster.group_r500()

    print('cop\t', cop)
    print('com\t', com)
    print('r500\t', r500)


if __name__ == "__main__":

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')
    else:
        main()

