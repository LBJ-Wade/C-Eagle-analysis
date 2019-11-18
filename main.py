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

    from cluster import Cluster
    from rendering import Map
    from mergers import dynamical_index, thermal_index
    from matplotlib import pyplot as plt
    import numpy as np
    import map_plot_parameters as plotpar

    plotpar.set_defaults_plot()

    # ceagle = Simulation()
    # z_catalogue = ceagle.get_redshiftAllowed(dtype = float)


    cluster = Cluster(clusterID = 4, redshift = 0.101)
    print('clusterID: ', cluster.clusterID)
    print('\tdynamical_index: ', dynamical_index(cluster))
    print('\tthermal_index: ', thermal_index(cluster))
    print('\n')

    CoP = cluster.group_centre_of_potential()
    CoM, _ = cluster.group_centre_of_mass(out_allPartTypes = False)
    coords = cluster.particle_coordinates('0')
    coords = np.subtract(coords, CoP)
    coords = cluster.comoving_length(coords)

    special_markers = np.vstack((CoP, CoM))
    special_markers = np.subtract(special_markers, CoP)
    special_markers_labels = [r'CoP', r'CoM']

    r500 = cluster.group_r200()
    r500 = cluster.comoving_length(r500)

    mass = cluster.particle_masses('0')
    mass = cluster.comoving_mass(mass)

    particles_map = Map()
    particles_map.xyz_projections(xyzdata = coords,
                                  weights = mass,
                                  plot_limit = 5*r500,
                                  nbins = 100,
                                  circle_pars = (0, 0, 1.5*r500),
                                  special_markers = special_markers,
                                  special_markers_labels = special_markers_labels)
    plt.show()


if __name__ == "__main__":

    import datetime

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')

    else:
        print(__FILE__)
        main()

