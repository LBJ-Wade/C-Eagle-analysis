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
    from mergers import dynamical_index, thermal_index

    # ceagle = Simulation()
    # z_catalogue = ceagle.get_redshiftAllowed(dtype = float)


    cluster = Cluster(clusterID = 0, redshift = 0.101)
    print('clusterID: ', cluster.clusterID)
    print('\tdynamical_index: ', dynamical_index(cluster))
    print('\tthermal_index: ', thermal_index(cluster))
    print('\n')

    x = cluster.particle_coordinates('0')[:,0]
    y = cluster.particle_coordinates('0')[:,1]
    z = cluster.particle_coordinates('0')[:,2]
    from matplotlib import pyplot as plt
    plt.scatter(x,y, marker=',', edgecolors='k')
    plt.show()


if __name__ == "__main__":

    import datetime

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')

    else:
        print(__FILE__)
        main()

