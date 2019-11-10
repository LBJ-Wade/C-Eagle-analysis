"""
main.py

AUTHOR: Edo Altamura
DATE: 10-11-2019

This file contains the main() function and is used for testing
purposes. It can also be linked to shell arguments for profiling.
By setting __PROFILE__ = False you are choosing to run the
program with normal outputs, while __PROFILE__ = True will trigger
the profiler and display the call stats associated with main().
"""

__PROFILE__ = False

def main():

    from clusters_retriever import Cluster

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

