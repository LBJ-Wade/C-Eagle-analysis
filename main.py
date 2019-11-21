__FILE__ = """------------------------------------------------------------------
FILE:   main.py
AUTHOR: Edo Altamura
DATE:   10-11-2019
------------------------------------------------------------------
This file contains the main() function and is used for testing
purposes. It can also be linked to shell arguments for profiling.
By setting __PROFILE__ = False you are choosing to run the
program with normal outputs, while __PROFILE__ = True will trigger
the profiler and display the call stats associated with main().
-------------------------------------------------------------------"""

__PROFILE__ = False

def time_func(function):
    # create a new function based on the existing one,
    # that includes the new timing behaviour
    def new_func(*args, **kwargs):
        start = datetime.datetime.now()
        print('Start: {}'.format(start))

        function_result = function(*args, **kwargs)
        # Calculate the elapsed time and add it to the function
        # attributes.
        end = datetime.datetime.now()
        new_func.elapsed = end - start
        print('End: {}'.format(end))
        print('Elapsed: {}'.format(new_func.elapsed))
        return function_result
    return new_func

@time_func
def main():

    from cluster import Cluster, Simulation
    from rendering import Map
    from mergers import dynamical_index, thermal_index
    import map_plot_parameters as plotpar
    from testing.angular_momentum import angular_momentum_PartType_alignment_matrix, alignment_DM_to_gas, \
        alignment_DM_to_stars, alignment_stars_to_gas

    from matplotlib import pyplot as plt
    import numpy as np


    plotpar.set_defaults_plot()

    sim = Simulation()
    # z_catalogue = ceagle.get_redshiftAllowed(dtype = float)

    ID = dyn_idx = th_idx = angmmo = dm2stars = dm2gas = stars2gas = []

    for i in range(0, 30):

        cluster = Cluster(clusterID = i, redshift = 0.101)
        angmom, masses = cluster.group_angular_momentum(out_allPartTypes=True)
        m = angular_momentum_PartType_alignment_matrix(cluster)

        print('clusterID: ', cluster.clusterID)
        print('\tdynamical_index: ', dynamical_index(cluster))
        print('\tthermal_index: ', thermal_index(cluster))
        print('\tangular momentum:', angmom)
        print('alignment_DM_to_gas\t', alignment_DM_to_gas(m))
        print('alignment_DM_to_stars\t', alignment_DM_to_stars(m))
        print('alignment_stars_to_gas\t', alignment_stars_to_gas(m))
        print('\n')

        ID.append(i)
        dyn_idx.append(dynamical_index(cluster))
        th_idx.append(thermal_index(cluster))
        angmmo.append(angmom)
        dm2stars.append(alignment_DM_to_gas(m))
        dm2gas.append(alignment_DM_to_stars(m))
        stars2gas.append(alignment_stars_to_gas(m))

    plt.scatter(np.array(ID), np.array(dm2gas))
    plt.show()


if __name__ == "__main__":

    import datetime

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')

    else:
        print(__FILE__)
        main()

