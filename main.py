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

    for i in [0]:
        cluster = Cluster(clusterID = i, redshift = 0.101)
        print('clusterID: ', cluster.clusterID)
        # print('group_centre_of_potential: ', cluster.group_centre_of_potential())
        # print('group_centre_of_mass: ', cluster.group_centre_of_mass())
        # print('dynamical_index: ', dynamical_index(cluster))
        print('thermal_index: ', thermal_index(cluster))


if __name__ == "__main__":

    import datetime

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')

    else:
        print(__FILE__)
        main()

