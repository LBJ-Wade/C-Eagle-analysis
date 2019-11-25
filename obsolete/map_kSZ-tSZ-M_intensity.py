import map_plot_parameters as plotpar
from obsolete import map_renderer_SZ as rendersz

import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists

# Turn off FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def map_kSZ_tSZ_2x1(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'show', projection = 0):
    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))

    rendersz.render_kSZ(axes[0], num_halo, redshift, simulation_type, projection=projection, nbins=nbins, rfov=rfov)
    rendersz.render_tSZ(axes[1], num_halo, redshift, simulation_type, projection=projection, nbins=nbins, rfov=rfov)

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        dir_name = 'Maps_kSZ+tSZ+M_intensity'
        save_name = 'map_kSZ+tSZ_2x1' + '_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name + '.pdf')


    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)


def map_M_kSZ_tSZ_3x1(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'save', projection = 0):
    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

    rendersz.render_M(axes[0], num_halo, redshift, simulation_type, projection=projection, nbins=nbins, rfov=rfov)
    rendersz.render_kSZ(axes[1], num_halo, redshift, simulation_type, projection=projection, nbins=nbins, rfov=rfov)
    rendersz.render_tSZ(axes[2], num_halo, redshift, simulation_type, projection=projection, nbins=nbins, rfov=rfov)

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        dir_name = 'Maps_M+kSZ+tSZ_3x1'
        save_name = 'map_M+kSZ+tSZ_3x1' + '_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name + '.pdf')

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

def call_maps(rank):
    num_halo = 11
    simulation_type = 'gas'
    redshift = 0.57

    # Call function:
    for num_halo in [0]:
        map_M_kSZ_tSZ_3x1(num_halo, redshift, simulation_type,
                          nbins = 600,
                          rfov = 5,
                          output = 'save',
                          projection = 0)
        print('completed' + str(num_halo))

# **************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_kSZ-tSZ-M_intensity.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    call_maps(rank)

# **************************************************************************************************