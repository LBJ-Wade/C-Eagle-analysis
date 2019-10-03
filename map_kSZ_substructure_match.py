from os import makedirs
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np

import catalog_compare as catcomp
import cluster_profiler as profile
import clusters_retriever as extract
import distance_cosmology as cosmo
import map_plot_parameters as plotpar
import map_renderer_SZ as maprender
import subhalo_marker as sgmark
import subhalo_selection as subsel
from bloblog_sub_id import substructure_identification
import gas_fractions_gridSpec_calculator as data

def select():
    temp_dict = {'SELECT_R_MIN': 0.9,
                  'SELECT_R_MAX': 5,

                  'SELECT_M_MIN': 10 ** 11.5,
                  'SELECT_M_MAX': 10 ** 14.5,

                  'SELECT_Fg_MIN': 0.02,
                  'SELECT_Fg_MAX': 0.55,

                  'SELECT_Vr_MIN': 200,
                  'SELECT_Vr_MAX': np.inf,

                  'SELECT_MV_MIN': -np.inf,
                  'SELECT_MV_MAX': np.inf}

    return temp_dict
def get_cat_data(halo_number, redshift, projection, sg_index):
    alldata_dict = {'index': data.std_HighRes_index(halo_number, redshift),
                    'H': halo_number * np.ones_like(data.std_HighRes_index(halo_number, redshift)),
                 # Index catalog from the hdf5 file (selected for high res region)
                 'R': data.std_r(halo_number, redshift),  # Units: physical R/R_200
                 'M': data.std_m_tot(halo_number, redshift),  # Units: physical solar masses
                 'Fg': data.std_gasFrac(halo_number, redshift),
                 'Vr': data.std_Vr(halo_number, redshift),  # Remember: this is a tuple of x,y,z coordinates
                 'MV': data.std_MVr(halo_number, redshift)}  # Remember: this is a tuple of x,y,z coordinates

    thirdaxis = [2, 1, 0]
    alldata_dict['Vr'] = alldata_dict['Vr'][thirdaxis[projection]]
    alldata_dict['MV'] = alldata_dict['MV'][thirdaxis[projection]]

    aux_i = [list(alldata_dict['index']).index(sg_index[i]) for i in range(len(sg_index))]

    # aux_i = np.concatenate([np.where(alldata_dict['index']==sgi)[0] for sgi in sg_index])

    cat_data = {'H': 0, 'I': 0,	'R': 0,	'M': 0,	'Fg': 0, 'Vr': 0, 'MV': 0}
    cat_data['H'] = alldata_dict['H'][aux_i].astype(np.int32)
    cat_data['I'] = alldata_dict['index'][aux_i].astype(np.int32)
    cat_data['R'] = alldata_dict['R'][aux_i]
    cat_data['M'] = alldata_dict['M'][aux_i]
    cat_data['Fg'] = alldata_dict['Fg'][aux_i]
    cat_data['Vr'] = alldata_dict['Vr'][aux_i]
    cat_data['MV'] = alldata_dict['MV'][aux_i]

    return cat_data

def get_cat_matched(redshift = 0.57, simulation_type = 'gas', projection = 0, nbins = 600, rfov = 5, norm = False, max_HaloNum = 10):
    cat_matched = {'HaloNum' : [], 'I' : []}
    master_cat_subfind = {'H': [], 'I': [],	'R': [],	'M': [],	'Fg': [], 'Vr': [], 'MV': []}
    for num_halo in range(max_HaloNum):
        print('matching halo ' + str(num_halo))
        path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
        file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

        r200 = extract.group_r200(path, file)
        h = extract.file_hubble_param(path, file)
        r200 = profile.comoving_length(r200, h, redshift)

        ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

        # Convert to angular distances
        angular_distance = cosmo.angular_diameter_D(redshift)
        Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
        r200 = r200 * Mpc_to_arcmin

        kSZmap = maprender.render_kSZ(None, num_halo, redshift, simulation_type, projection=projection, nbins=nbins,
                                      rfov=rfov)
        cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200)
        selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
        sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
        cat_subfind = {
            'X': sg_xyz[ijk[projection][0]],
            'Y': sg_xyz[ijk[projection][1]],
            'I': selected_dict['I']
        }
        subs_within = catcomp.compare(cat_subfind, cat_blobs)

        idx = np.concatenate([np.where(cat_subfind['I'] == subs_within[j])[0] for j in range(len(subs_within)) if
                              not np.isnan(subs_within[j])])
        idx2 = np.unique([int(i) for i in subs_within if not np.isnan(i)])

        cat_matched['HaloNum'].append(num_halo*np.ones_like(idx2))
        cat_matched['I'].append(idx2)

        if norm:
            for key in master_cat_subfind.keys():
                master_cat_subfind[key].append(selected_dict[key])
        pass


    cat_matched['HaloNum'] = np.concatenate(cat_matched['HaloNum'])
    cat_matched['I'] = np.concatenate(cat_matched['I'])


    if not norm:
        return cat_matched
    else:
        for key in master_cat_subfind.keys():
            master_cat_subfind[key]=np.concatenate(master_cat_subfind[key])
        return cat_matched, master_cat_subfind

def write_cats_ByHalo(redshift = 0.57, simulation_type = 'gas', projection = 0, nbins = 600, rfov = 5, norm = True, max_HaloNum = 10):
    """
    Usage: CMD  >> from map_kSZ_substructure_match import write_cats_ByHalo as wcbh
                >> wcbh(max_HaloNum = <maximum halo number>)
    """
    dir_name = 'Substructure_Match_Output_CMBrestFrame'
    if not exists(dir_name): makedirs(dir_name)

    for num_halo in range(max_HaloNum):
        print('matching halo ' + str(num_halo))
        path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
        file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

        r200 = extract.group_r200(path, file)
        h = extract.file_hubble_param(path, file)
        r200 = profile.comoving_length(r200, h, redshift)

        ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

        # Convert to angular distances
        angular_distance = cosmo.angular_diameter_D(redshift)
        Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
        r200 = r200 * Mpc_to_arcmin

        kSZmap = maprender.render_kSZ(None, num_halo, redshift, simulation_type, projection=projection, nbins=nbins,
                                      rfov=rfov)
        cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200)
        selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
        sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
        cat_subfind = {
            'X': sg_xyz[ijk[projection][0]],
            'Y': sg_xyz[ijk[projection][1]],
            'I': selected_dict['I']
        }
        subs_within = catcomp.compare(cat_subfind, cat_blobs)

        idx2 = np.unique([int(i) for i in subs_within if not np.isnan(i)])

        matched_dict = get_cat_data(num_halo, redshift, projection, idx2)

        save_name_matched = 'matched_kSZ' + '_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)
        save_name_selected = 'selected_kSZ' + '_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        np.save(dir_name + '//' + save_name_matched + '.npy', matched_dict)
        np.save(dir_name + '//' + save_name_selected + '.npy', selected_dict)



def map_kSZ_substructurematch(num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, output = 'show'):
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

    r200 = extract.group_r200(path, file)
    h = extract.file_hubble_param(path, file)
    r200 = profile.comoving_length(r200, h, redshift)

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    r200 = r200 * Mpc_to_arcmin

    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

    # First subplot: blob detection

    # This function returns the map but also plots it on the axis using the fact that object axes is mutable
    kSZmap = maprender.render_kSZ(axes[2], num_halo, redshift, simulation_type, projection = projection, nbins = nbins, rfov=rfov)

    cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200, plotmap=True, axes=axes[0])
    for k in range(len(cat_blobs['I'])):
        circ = plt.Circle((cat_blobs['X'][k], cat_blobs['Y'][k]), cat_blobs['R'][k], color='red', linewidth=1, fill=False)
        axes[0].add_patch(circ)

    # Second subplot: marker catalog comparison and matching
    selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
    sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
    cat_subfind = {
        'X': sg_xyz[ijk[projection][0]],
        'Y': sg_xyz[ijk[projection][1]],
        'I': selected_dict['I']
    }
    subs_within = catcomp.compare(cat_subfind, cat_blobs)
    catcomp.render_figure(axes[1], cat_blobs, cat_subfind, subs_within, annotate=False)

    # Third subplot: matched markers only
    idx = np.concatenate([np.where(cat_subfind['I'] == subs_within[j])[0] for j in range(len(subs_within)) if not np.isnan(subs_within[j])])
    idx2 = np.unique([int(i) for i in subs_within if not np.isnan(i)])
    print(idx2)

    axes[2].scatter(cat_subfind['X'][idx], cat_subfind['Y'][idx], s=20, marker='x', facecolors='k', label = r'$\mathrm{Identified\ subhalos}$')
    #axes[2].legend(loc = 'lower right', fontsize = 18)

    if output.lower() == "show":
        plt.show()

    if output.lower() == "save":
        dir_name = 'Substructure_Match_Process'
        save_name = 'sg-match-process-kSZ' + '_new_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        if not exists(dir_name): makedirs(dir_name)

        #plt.savefig(dir_name + '//' + save_name + '_400dpi' + '.png', dpi=400)
        #plt.savefig(dir_name + '//' + save_name + '_600dpi' + '.png', dpi=600)
        plt.savefig(dir_name + '//' + save_name + '.pdf')
        np.save(dir_name + '//' + save_name + '.npy', idx2)


def map_kSZ_substructurematch_1x(num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, output = 'show'):
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

    r200 = extract.group_r200(path, file)
    h = extract.file_hubble_param(path, file)
    r200 = profile.comoving_length(r200, h, redshift)

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    r200 = r200 * Mpc_to_arcmin

    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    # First subplot: blob detection

    # This function returns the map but also plots it on the axis using the fact that object axes is mutable
    kSZmap = maprender.render_kSZ(None, num_halo, redshift, simulation_type, projection = projection, nbins = nbins, rfov=rfov)

    cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200, plotmap=True, axes=axes)
    for k in range(len(cat_blobs['I'])):
        circ = plt.Circle((cat_blobs['X'][k], cat_blobs['Y'][k]), cat_blobs['R'][k], color='red', linewidth=1, fill=False)
        axes.add_patch(circ)

    # Second subplot: marker catalog comparison and matching
    selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
    sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
    cat_subfind = {
        'X': sg_xyz[ijk[projection][0]],
        'Y': sg_xyz[ijk[projection][1]],
        'I': selected_dict['I']
    }
    subs_within = catcomp.compare(cat_subfind, cat_blobs)
    #catcomp.render_figure(None, cat_blobs, cat_subfind, subs_within, annotate=False)

    # Third subplot: matched markers only
    idx = np.concatenate([np.where(cat_subfind['I'] == subs_within[j])[0] for j in range(len(subs_within)) if not np.isnan(subs_within[j])])
    idx2 = np.unique([int(i) for i in subs_within if not np.isnan(i)])
    print(idx2)

    #axes[2].scatter(cat_subfind['X'][idx], cat_subfind['Y'][idx], s=20, marker='x', facecolors='k', label = r'$\mathrm{Identified\ subhalos}$')
    #axes[2].legend(loc = 'lower right', fontsize = 18)

    if output.lower() == "show":
        plt.show()

    if output.lower() == "save":
        dir_name = 'Substructure_Match_Process'
        save_name = 'sg-match-process-kSZ' + '_new1x_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        if not exists(dir_name): makedirs(dir_name)
        plt.tight_layout()
        #plt.savefig(dir_name + '//' + save_name + '_400dpi' + '.png', dpi=400)
        #plt.savefig(dir_name + '//' + save_name + '_600dpi' + '.png', dpi=600)
        plt.savefig(dir_name + '//' + save_name + '.png')
        np.save(dir_name + '//' + save_name + '.npy', idx2)


def map_kSZ_substructurematch_2x(num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, output = 'show'):
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

    r200 = extract.group_r200(path, file)
    h = extract.file_hubble_param(path, file)
    r200 = profile.comoving_length(r200, h, redshift)

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    r200 = r200 * Mpc_to_arcmin

    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 9))

    # First subplot: blob detection

    # This function returns the map but also plots it on the axis using the fact that object axes is mutable
    kSZmap = maprender.render_kSZ(axes[1], num_halo, redshift, simulation_type, projection = projection, nbins = nbins, rfov=rfov)

    cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200, plotmap=True, axes=None)
    for k in range(len(cat_blobs['I'])):
        circ = plt.Circle((cat_blobs['X'][k], cat_blobs['Y'][k]), cat_blobs['R'][k], color='red', linewidth=1, fill=False)
        #axes[0].add_patch(circ)

    # Second subplot: marker catalog comparison and matching
    selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
    sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
    cat_subfind = {
        'X': sg_xyz[ijk[projection][0]],
        'Y': sg_xyz[ijk[projection][1]],
        'I': selected_dict['I']
    }
    subs_within = catcomp.compare(cat_subfind, cat_blobs)
    catcomp.render_figure(axes[0], cat_blobs, cat_subfind, subs_within, annotate=False)

    # Third subplot: matched markers only
    idx = np.concatenate([np.where(cat_subfind['I'] == subs_within[j])[0] for j in range(len(subs_within)) if not np.isnan(subs_within[j])])
    idx2 = np.unique([int(i) for i in subs_within if not np.isnan(i)])
    print(idx2)

    axes[1].scatter(cat_subfind['X'][idx], cat_subfind['Y'][idx], s=20, marker='x', facecolors='k', label = r'$\mathrm{Identified\ subhalos}$')
    #axes[2].legend(loc = 'lower right', fontsize = 18)

    if output.lower() == "show":
        plt.show()

    if output.lower() == "save":
        dir_name = 'Substructure_Match_Process'
        save_name = 'sg-match-process-kSZ' + '_new_halo2x' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        if not exists(dir_name): makedirs(dir_name)
        plt.tight_layout()

        #plt.savefig(dir_name + '//' + save_name + '_400dpi' + '.png', dpi=400)
        #plt.savefig(dir_name + '//' + save_name + '_600dpi' + '.png', dpi=600)
        plt.savefig(dir_name + '//' + save_name + '.pdf')
        np.save(dir_name + '//' + save_name + '.npy', idx2)

def map_kSZ_greyblob(num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, output = 'show'):
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

    r200 = extract.group_r200(path, file)
    h = extract.file_hubble_param(path, file)
    r200 = profile.comoving_length(r200, h, redshift)

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    r200 = r200 * Mpc_to_arcmin

    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    kSZmap = maprender.render_kSZ(None, num_halo, redshift, simulation_type, projection=projection, nbins=nbins,
                                  rfov=rfov)

    cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200, plotmap=True, axes=axes)
    for k in range(len(cat_blobs['I'])):
        circ = plt.Circle((cat_blobs['X'][k], cat_blobs['Y'][k]), cat_blobs['R'][k], color='fuchsia', linewidth=1,
                          fill=False)
        axes.add_patch(circ)

    if output.lower() == "show":
        plt.show()

    if output.lower() == "save":
        dir_name = 'Substructure_Match_Process//substructure_identification_only_1x1'
        save_name = 'sub-id_white_halo' + str(num_halo)

        if not exists(dir_name): makedirs(dir_name)
        plt.tight_layout()
        print(dir_name + '//' + save_name + '_400dpi' + '.png')
        #plt.savefig(dir_name + '//' + save_name + '_400dpi' + '.png', dpi=400)
        #plt.savefig(dir_name + '//' + save_name + '_600dpi' + '.png', dpi=600)
        plt.savefig(dir_name + '//' + save_name + '.png', facecolor=fig.get_facecolor(), edgecolor='none')

def map_kSZ_match(num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, output = 'show'):
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))

    r200 = extract.group_r200(path, file)
    h = extract.file_hubble_param(path, file)
    r200 = profile.comoving_length(r200, h, redshift)

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    r200 = r200 * Mpc_to_arcmin

    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    kSZmap = maprender.render_kSZ(None, num_halo, redshift, simulation_type, projection = projection, nbins = nbins, rfov=rfov)

    cat_blobs = substructure_identification(np.asarray(kSZmap), rfov, r200, plotmap=True, axes=None)

    selected_dict = subsel.compute_catalog(num_halo, redshift, projection, output="Return", default_select=False, set_select=select())
    sg_xyz = sgmark.subhalo_marks_filtered(path, file, selected_dict['I'])
    cat_subfind = {
        'X': sg_xyz[ijk[projection][0]],
        'Y': sg_xyz[ijk[projection][1]],
        'I': selected_dict['I']
    }
    subs_within = catcomp.compare(cat_subfind, cat_blobs)
    catcomp.render_figure(axes, cat_blobs, cat_subfind, subs_within, annotate=False)

    if output.lower() == "show":
        plt.show()

    if output.lower() == "save":
        dir_name = 'Substructure_Match_Process//substructure_match_only_1x1'
        save_name = 'sub-match_white_halo' + str(num_halo)

        if not exists(dir_name): makedirs(dir_name)
        plt.tight_layout()
        print(dir_name + '//' + save_name + '_400dpi' + '.png')
        #plt.savefig(dir_name + '//' + save_name + '_400dpi' + '.png', dpi=400)
        #plt.savefig(dir_name + '//' + save_name + '_600dpi' + '.png', dpi=600)
        plt.savefig(dir_name + '//' + save_name + '.png', facecolor=fig.get_facecolor(), edgecolor='none')
#**************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_kSZ_substructure_match.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    for num_halo in range(1):
        map_kSZ_match(num_halo, 0.57, 'gas', projection=0, nbins=600, rfov=5, output='save')
        map_kSZ_greyblob(num_halo, 0.57, 'gas', projection=0, nbins=600, rfov=5, output='save')
    #map_kSZ_substructurematch_1x(0, 0.57, 'gas', nbins = 600, output='save')
    #map_kSZ_substructurematch_2x(0, 0.57, 'gas', nbins = 600, output='save')