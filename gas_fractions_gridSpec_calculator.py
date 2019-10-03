import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen

import numpy as np

def get_PathFile(num_halo, redshift):
	path = extract.path_from_cluster_name(num_halo, simulation_type = 'gas')
	file = extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
	return path, file

def get_sgn(num_halo, redshift):
	path, file = get_PathFile(num_halo, redshift)
	group_number = extract.subgroups_group_number(path, file)
	index = np.where(group_number>-1)[0]
	return index

def get_r(num_halo, redshift):
	path, file = get_PathFile(num_halo, redshift)
	h = extract.file_hubble_param(path, file)
	r200 = extract.group_r200(path, file)
	group_CoP = extract.group_centre_of_potential(path, file)
	coordinates = extract.subgroups_centre_of_mass(path, file)
	x = coordinates[:,0] - group_CoP[0]
	y = coordinates[:,1] - group_CoP[1]
	z = coordinates[:,2] - group_CoP[2]
	r200 = profile.comoving_length(r200, h, redshift)
	x = profile.comoving_length(x, h, redshift)
	y = profile.comoving_length(y, h, redshift)
	z = profile.comoving_length(z, h, redshift)
	return np.sqrt(x**2+y**2+z**2)/r200 # Units: physical R/R_200

def get_gasFrac(num_halo, redshift):
	path, file = get_PathFile(num_halo, redshift)
	h = extract.file_hubble_param(path, file)
	redshift = extract.file_redshift(path, file)

	mass_tot = extract.subgroups_mass(path, file)
	mass_tot = profile.comoving_mass(mass_tot, h, redshift)
	mass_types = extract.subgroups_mass_type(path, file)
	mass_types = profile.comoving_mass(mass_types, h, redshift)
	f_gas = mass_types[:, 0] / mass_tot[:]
	return f_gas

def get_m_tot(num_halo, redshift, units = 'astro'):
	path, file = get_PathFile(num_halo, redshift)
	h = extract.file_hubble_param(path, file)
	redshift = extract.file_redshift(path, file)

	mass_tot = extract.subgroups_mass(path, file)
	mass_tot = profile.comoving_mass(mass_tot, h, redshift)
	mass_tot = profile.mass_units(mass_tot, unit_system = units)
	return mass_tot.astype(np.longdouble) # Units: physical solar masses

def get_Vr(num_halo, redshift, units = 'astro'):
	path, file = get_PathFile(num_halo, redshift)
	file_part = extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
	tot_rest_frame, _ = profile.total_mass_rest_frame(path, file_part)
	h = extract.file_hubble_param(path, file)
	velocities = extract.subgroups_velocity(path, file)
	vx = velocities[:,0] - tot_rest_frame[0]
	vy = velocities[:,1] - tot_rest_frame[1]
	vz = velocities[:,2] - tot_rest_frame[2]
	vx = profile.comoving_velocity(vx, h, redshift)
	vy = profile.comoving_velocity(vy, h, redshift)
	vz = profile.comoving_velocity(vz, h, redshift)    
	vx = profile.velocity_units(vx, unit_system = units)
	vy = profile.velocity_units(vy, unit_system = units)
	vz = profile.velocity_units(vz, unit_system = units)
	return vx.astype(np.longdouble).astype(np.longdouble), vy.astype(np.longdouble).astype(np.longdouble), vz.astype(np.longdouble).astype(np.longdouble)

def get_MVr(num_halo, redshift, units = 'SI'):
	mass_tot = get_m_tot(num_halo, redshift, units = units)
	vx, vy, vz = get_Vr(num_halo, redshift, units = units)
	MVx, MVy, MVz = vx*mass_tot, vy*mass_tot, vz*mass_tot
	MVx, MVy, MVz = MVx.astype(np.longdouble).astype(np.longdouble), MVy.astype(np.longdouble), MVz.astype(np.longdouble)
	return MVx, MVy, MVz

# Standard data retriver - standard units - 5R200 selection
def std_gasFrac(num_halo, redshift):
	sgn = get_sgn(num_halo, redshift)
	r = get_r(num_halo, redshift)
	fg = get_gasFrac(num_halo, redshift)
	index = np.where(r<5)[0]
	fg = fg[index]
	return fg # Return units: dimensionless

def std_m_tot(num_halo, redshift):
	sgn = get_sgn(num_halo, redshift)
	r = get_r(num_halo, redshift)
	m = get_m_tot(num_halo, redshift, units = 'astro')
	index = np.where(r<5)[0]
	m = m[index]
	return m # Return units: solar masses

def std_r(num_halo, redshift):
	sgn = get_sgn(num_halo, redshift)
	r = get_r(num_halo, redshift)
	index = np.where(r<5)[0]
	r = r[index]
	return r # Returnunits: dimensionless

def std_Vr(num_halo, redshift):
	sgn = get_sgn(num_halo, redshift)
	r = get_r(num_halo, redshift)
	Vx, Vy, Vz = get_Vr(num_halo, redshift, units = 'astro')
	index = np.where(r<5)[0]
	Vx, Vy, Vz = Vx[index], Vy[index], Vz[index]
	return Vx, Vy, Vz # Return units: km/s

def std_MVr(num_halo, redshift):
	sgn = get_sgn(num_halo, redshift)
	r = get_r(num_halo, redshift)
	MVx, MVy, MVz = get_MVr(num_halo, redshift, units = 'astro')
	index = np.where(r<5)[0]
	MVx, MVy, MVz = MVx[index], MVy[index], MVz[index]
	return MVx, MVy, MVz # Return units: kg m/s

def std_HighRes_index(num_halo, redshift):
	r = get_r(num_halo, redshift)
	index = np.where(r<5)[0]
	return index