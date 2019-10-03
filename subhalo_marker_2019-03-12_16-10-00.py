import clusters_retriever as extract
import cluster_profiler as profile

import numpy as np
#import numba
import astropy
from astropy.cosmology import FlatLambdaCDM
import distance_cosmology as cosmo

#@numba.jit(nopython = True, parallel = True)
def subhalo_marks(path, file, momentum_threshold):
    # LOAD FOF GROUP DATA
    group_CoP = extract.group_centre_of_potential(path, file)
    r200 = extract.group_r200(path, file)

    # LOAD SUBHALO DATA
    part_type = extract.particle_type('gas')
    subhalo_cop = extract.subgroups_centre_of_potential(path, file)
    subhalo_gn = extract.subgroups_group_number(path, file)
    subhalo_momentum = profile.subhalo_average_momentum(path, file, part_type)

    # CONSTANTS
    h = extract.file_hubble_param(path, file)
    z = extract.file_redshift(path, file)

    # RELATIVE COORDINATES
    sgx = subhalo_cop[:, 0] - group_CoP[0]
    sgy = subhalo_cop[:, 1] - group_CoP[1]
    sgz = subhalo_cop[:, 2] - group_CoP[2]

    # COMOVING COORDINATES
    r200 = profile.comoving_length(r200, h, z)
    sgx = profile.comoving_length(sgx, h, z)
    sgy = profile.comoving_length(sgy, h, z)
    sgz = profile.comoving_length(sgz, h, z)
    sgr = np.sqrt(sgx ** 2 + sgy ** 2 + sgz ** 2)

    # subgroup momentum sgp_ - trailing underscore means this is vector (2D array)
    sgp_ = profile.comoving_momentum(subhalo_momentum, h, z)

    # SUBHALO SELECTION
    # - coordinate based - distance from FOF centre
    sgindex = np.where(sgr < 5*r200)[0]
    sgx, sgy, sgz = sgx[sgindex], sgy[sgindex], sgz[sgindex]
    sgp_ = sgp_[sgindex, :]
    sggn = subhalo_gn[sgindex]

    # - momentum based - separate for each projection
    # Output selected phase space data of subhalos to 2D array containing
    #   position in 1st coordinate and 2nd coordinate, momentum in 3rd coordinate
    sgindex_xy = np.where(abs(sgp_[:, 2]) > momentum_threshold*np.median(abs(sgp_[:, 2])))[0]
    sg_x_y_pz_gn = np.ascontiguousarray([sgx[sgindex_xy], sgy[sgindex_xy], sgp_[sgindex_xy, 2], sggn[sgindex_xy]])

    sgindex_yz = np.where(abs(sgp_[:, 0]) > momentum_threshold * np.median(abs(sgp_[:, 0])))[0]
    sg_y_z_px_gn = np.asarray([sgy[sgindex_yz], sgz[sgindex_yz], sgp_[sgindex_yz, 0], sggn[sgindex_yz]])

    sgindex_xz = np.where(abs(sgp_[:, 1]) > momentum_threshold * np.median(abs(sgp_[:, 1])))[0]
    sg_x_z_py_gn = np.asarray([sgx[sgindex_xz], sgz[sgindex_xz], sgp_[sgindex_xz, 1], sggn[sgindex_xz]])

    # Convert to angular coordinates
    angular_distance = cosmo.angular_diameter_D(z)
    Mpc_to_arcmin = np.power(np.pi, -1)*180*60/angular_distance

    print(angular_distance)
    print(Mpc_to_arcmin)

    sg_x_y_pz_gn[0:2, :] *= Mpc_to_arcmin
    sg_y_z_px_gn[0:2, :] *= Mpc_to_arcmin
    sg_x_z_py_gn[0:2, :] *= Mpc_to_arcmin

    # Calculate marker sizes
    ms_pz = np.expand_dims(marksize(sg_x_y_pz_gn[2, :]), axis=0)
    ms_px = np.expand_dims(marksize(sg_y_z_px_gn[2, :]), axis=0)
    ms_py = np.expand_dims(marksize(sg_x_z_py_gn[2, :]), axis=0)

    # Append marker sizes
    sg_x_y_pz_gn_ms = np.append(sg_x_y_pz_gn, ms_pz, axis=0)
    sg_y_z_px_gn_ms = np.append(sg_y_z_px_gn, ms_px, axis=0)
    sg_x_z_py_gn_ms = np.append(sg_x_z_py_gn, ms_py, axis=0)

    # Generate final array
    sg_final = np.stack((sg_x_y_pz_gn_ms, sg_y_z_px_gn_ms, sg_x_z_py_gn_ms), axis=2)

    return sg_final

#@numba.jit(nopython = True, parallel = True)
def marksize(q):
    q[q==0] = np.nan
    a = 10
    b = 2
    s = a*np.log10(np.abs(q/np.max(q)))+b
    return s
