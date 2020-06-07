import sys
import os
import itertools
import argparse
from typing import List, Dict, Union
from mpi4py import MPI
import numpy as np
import h5py as h5
import datetime
from scipy.sparse import csr_matrix

from .__init__ import pprint
from .conversion import (
	comoving_density,
	comoving_length,
	comoving_velocity,
	comoving_mass,
	comoving_kinetic_energy,
	comoving_momentum,
	comoving_ang_momentum,
	density_units,
	velocity_units,
	length_units,
	mass_units,
	momentum_units,
	energy_units,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def split(nfiles):
    nfiles=int(nfiles)
    nf=int(nfiles/nproc)
    rmd=nfiles % nproc
    st=rank*nf
    fh=(rank+1)*nf
    if rank < rmd:
        st+=rank
        fh+=(rank+1)
    else:
        st+=rmd
        fh+=rmd
    return st,fh

def commune(data):
    tmp=np.zeros(nproc,dtype=np.int)
    tmp[rank]=len(data)
    cnts=np.zeros(nproc,dtype=np.int)
    comm.Allreduce([tmp,MPI.INT],[cnts,MPI.INT],op=MPI.SUM)
    del tmp
    dspl=np.zeros(nproc,dtype=np.int)
    i=0
    for j in range(0,nproc,1):
        dspl[j]=i
        i+=cnts[j]
    rslt=np.zeros(i,dtype=data.dtype)
    comm.Allgatherv([data,cnts[rank]],[rslt,cnts,dspl,MPI._typedict[data.dtype.char]])
    del data,cnts,dspl
    return rslt

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

def find_files(redshift: str):
    z_value = ['z003p000', 'z002p750', 'z002p500', 'z002p250', 'z002p000', 'z001p750', 'z001p500', 'z001p250',
     'z001p000', 'z000p750', 'z000p500', 'z000p375', 'z000p250', 'z000p125', 'z000p000']
    z_IDNumber = ['018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
     '031', '032']
    sn = dict(zip(z_value, z_IDNumber))[redshift]
    path='/scratch/nas_virgo/Cosmo-OWLS/AGN_TUNED_nu0_L400N1024_Planck'
    pprint(f"[+] Find simulation files {redshift:s}...")
    pd = ''
    if os.path.isfile(path+'/particledata_'+sn+'/eagle_subfind_particles_'+sn+'.0.hdf5'):
        pd=path+'/particledata_'+sn+'/eagle_subfind_particles_'+sn+'.0.hdf5'
    sd=[]
    for x in os.listdir(path+'/groups_'+sn+'/'):
        if x.startswith('eagle_subfind_tab_'):
            sd.append(path+'/groups_'+sn+'/'+x)
    odr=[]
    for x in sd:
        odr.append(int(x[94:-5]))
    odr=np.array(odr)
    so=np.argsort(odr)
    sd=list(np.array(sd)[so])
    return [sd,pd]

def fof_header(files: list):
	pprint(f"[+] Find header information...")
	header = {}
	with h5.File(files[1], 'r') as f:
		header['Hub']  = f['Header'].attrs['HubbleParam']
		header['aexp'] = f['Header'].attrs['ExpansionFactor']
		header['zred'] = f['Header'].attrs['Redshift']
		header['OmgL'] = f['Header'].attrs['OmegaLambda']
		header['OmgM'] = f['Header'].attrs['Omega0']
		header['OmgB'] = f['Header'].attrs['OmegaBaryon']
	return header

def fof_groups(files: list, header: Dict[str, float]):
	pprint(f"[+] Find group information for whole snapshot...")
	st, fh = split(len(files[0]))
	Mfof = np.empty(0)
	M2500 = np.empty(0)
	M500 = np.empty(0)
	M200 = np.empty(0)
	R2500 = np.empty(0)
	R500 = np.empty(0)
	R200 = np.empty(0)
	COP = np.empty(0)
	NSUB = np.empty(0)
	FSID = np.empty(0)
	SCOP = np.empty(0)
	for x in range(st, fh, 1):
		with h5.File(files[0][x], 'r') as f:
			Mfof = np.append(Mfof, f['FOF/GroupMass'][:])
			M2500 = np.append(M2500, f['FOF/Group_M_Crit2500'][:])
			R2500 = np.append(R2500, f['FOF/Group_R_Crit2500'][:])
			M500 = np.append(M500, f['FOF/Group_M_Crit500'][:])
			R500 = np.append(R500, f['FOF/Group_R_Crit500'][:])
			M200 = np.append(M200, f['FOF/Group_M_Crit200'][:])
			R200 = np.append(R200, f['FOF/Group_R_Crit200'][:])
			COP = np.append(COP, f['FOF/GroupCentreOfPotential'][:])
			NSUB = np.append(NSUB, f['FOF/NumOfSubhalos'][:])
			FSID = np.append(FSID, f['FOF/FirstSubhaloID'][:])
			SCOP = np.append(SCOP, f['Subhalo/CentreOfPotential'][:])

	data = {}
	data['Mfof'] = commune(Mfof)
	data['M2500'] = commune(M2500)
	data['R2500'] = commune(R2500)
	data['M500'] = commune(M500)
	data['R500'] = commune(R500)
	data['M200'] = commune(M200)
	data['R200'] = commune(R200)
	data['COP']  = commune(COP.reshape(-1, 1)).reshape(-1, 3)
	data['NSUB'] = commune(NSUB)
	data['FSID'] = commune(FSID)
	data['SCOP'] = commune(SCOP.reshape(-1, 1)).reshape(-1, 3)
	data['idx'] = np.where(M500 > 1.0e13)[0]

	# Conversion
	data['Mfof'] = comoving_mass(header, data['Mfof']* 1.0e10)
	data['M2500'] = comoving_mass(header, data['M2500']* 1.0e10)
	data['M500'] = comoving_mass(header, data['M500']* 1.0e10)
	data['M200'] = comoving_mass(header, data['M200']* 1.0e10)
	data['R2500'] = comoving_length(header, data['R2500'])
	data['R500'] = comoving_length(header, data['R500'])
	data['R200'] = comoving_length(header, data['R200'])
	data['COP']  = comoving_length(header, data['COP'])
	data['SCOP'] = comoving_length(header, data['SCOP'])

	return data

def fof_group(clusterID: int, fofgroups: Dict[str, np.ndarray] = None):
	pprint(f"[+] Find group information for cluster {clusterID}")
	new_data = {}
	new_data['clusterID'] = clusterID
	new_data['idx']   = fofgroups['idx'][clusterID]
	new_data['Mfof']  = fofgroups['Mfof'][new_data['idx']]
	new_data['M2500'] = fofgroups['M2500'][new_data['idx']]
	new_data['R2500'] = fofgroups['R2500'][new_data['idx']]
	new_data['M500']  = fofgroups['M500'][new_data['idx']]
	new_data['R500']  = fofgroups['R500'][new_data['idx']]
	new_data['M200']  = fofgroups['M200'][new_data['idx']]
	new_data['R200']  = fofgroups['R200'][new_data['idx']]
	new_data['COP']   = fofgroups['COP'][new_data['idx']]
	new_data['NSUB']  = fofgroups['NSUB'][new_data['idx']]
	new_data['FSID']  = fofgroups['FSID'][new_data['idx']]
	new_data['SCOP']  = fofgroups['SCOP'][new_data['idx']]
	return new_data


def snap_groupnumbers(files: list, fofgroups: Dict[str, np.ndarray] = None):
	halo_num_catalogue_contiguous = np.max(fofgroups['idx'])

	with h5.File(files[1], 'r') as h5file:
		Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0, 1, 4]]

		pprint(f"[+] Collecting gas particles GroupNumber...")
		st, fh = split(Nparticles[0])
		groupnumber0 = h5file[f'/PartType0/GroupNumber'][st:fh]
		# Clip out negative values and exceeding values
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber0 = np.clip(groupnumber0, 0, halo_num_catalogue_contiguous + 2)
		groupnumber0_csrm = get_indices_sparse(groupnumber0)
		del groupnumber0

		pprint(f"[+] Collecting CDM particles GroupNumber...")
		st, fh = split(Nparticles[1])
		groupnumber1 = h5file[f'/PartType1/GroupNumber'][st:fh]
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber1 = np.clip(groupnumber1, 0, halo_num_catalogue_contiguous + 2)
		groupnumber1_csrm = get_indices_sparse(groupnumber1)
		del groupnumber1

		pprint(f"[+] Collecting star particles GroupNumber...")
		st, fh = split(Nparticles[2])
		groupnumber4 = h5file[f'/PartType4/GroupNumber'][st:fh]
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber4 = np.clip(groupnumber4, 0, halo_num_catalogue_contiguous + 2)
		groupnumber4_csrm = get_indices_sparse(groupnumber4)
		del groupnumber4

	return [groupnumber0_csrm, groupnumber1_csrm, groupnumber4_csrm]


def cluster_particles(files: list, header: Dict[str, float], fofgroup: Dict[str, np.ndarray] = None, groupNumbers: List[np.ndarray] = None):
	pprint(f"[+] Find particle information for cluster {fofgroup['clusterID']}")
	with h5.File(files[1], 'r') as h5file:
		data_out = {}
		partTypes = ['0', '1', '4']
		Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0, 1, 4]]

		for pt in partTypes:

			# Gather groupnumbers from cores
			st, fh = split(Nparticles[partTypes.index(pt)])
			fof_id = fofgroup['idx'] + 1
			groupNumber = groupNumbers[partTypes.index(pt)][fof_id][0] + st
			pgn = commune(groupNumber)
			del groupNumber

			# Filter particle data with collected groupNumber indexing
			data_out[f'partType{pt}'] = {}
			data_out[f'partType{pt}']['subgroup_number'] = h5file[f'/PartType{pt}/SubGroupNumber'][pgn]
			data_out[f'partType{pt}']['velocity'] = h5file[f'/PartType{pt}/Velocity'][pgn]
			data_out[f'partType{pt}']['coordinates'] = h5file[f'/PartType{pt}/Coordinates'][pgn]
			if pt == '1':
				particle_mass_DM = h5file['Header'].attrs['MassTable'][1]
				data_out[f'partType{pt}']['mass'] = np.ones(len(pgn), dtype=np.float) * particle_mass_DM
			else:
				data_out[f'partType{pt}']['mass'] = h5file[f'/PartType{pt}/Mass'][pgn]
			if pt == '0':
				data_out[f'partType{pt}']['temperature'] = h5file[f'/PartType{pt}/Temperature'][pgn]
				data_out[f'partType{pt}']['sphdensity'] = h5file[f'/PartType{pt}/Density'][pgn]
				data_out[f'partType{pt}']['sphlength'] = h5file[f'/PartType{pt}/SmoothingLength'][pgn]

			# Conversion from comoving units to physical units
			data_out[f'partType{pt}']['velocity'] = comoving_velocity(header, data_out[f'partType{pt}']['velocity'])
			data_out[f'partType{pt}']['coordinates'] = comoving_length(header, data_out[f'partType{pt}']['coordinates'])
			data_out[f'partType{pt}']['mass'] = comoving_mass(header, data_out[f'partType{pt}']['mass'] * 1.0e10)
			if pt == '0':
				den_conv = h5file[f'/PartType{pt}/Density'].attrs['CGSConversionFactor']
				data_out[f'partType{pt}']['sphdensity'] = comoving_density(header, data_out[f'partType{pt}']['sphdensity'] * den_conv)
				data_out[f'partType{pt}']['sphlength'] = comoving_length(header, data_out[f'partType{pt}']['sphlength'])

			# Periodic boundary wrapping of particle coordinates
			boxsize = comoving_length(header, h5file['Header'].attrs['BoxSize'])
			coords = data_out[f'partType{pt}']['coordinates']
			for coord_axis in range(3):
				# Right boundary
				if fofgroup['COP'][coord_axis] + 5 * fofgroup['R200'] > boxsize:
					beyond_index = np.where(coords[:, coord_axis] < boxsize / 2)[0]
					coords[beyond_index, coord_axis] = coords[beyond_index, coord_axis] + boxsize
					del beyond_index
				# Left boundary
				elif fofgroup['COP'][coord_axis] - 5 * fofgroup['R200'] < 0.:
					beyond_index = np.where(coords[:, coord_axis] > boxsize / 2)[0]
					coords[beyond_index, coord_axis] = coords[beyond_index, coord_axis] - boxsize
					del beyond_index
			data_out[f'partType{pt}']['coordinates'] = coords
			del coords

	return data_out

def cluster_data(clusterID: int,
                 files: list,
                 header: Dict[str, float],
                 fofgroups: Dict[str, np.ndarray] = None,
                 groupNumbers: List[np.ndarray] = None):

	group_data = fof_group(clusterID, fofgroups = fofgroups)
	part_data = cluster_particles(files, header, fofgroup=group_data, groupNumbers= groupNumbers)

	out = {}
	out['Header'] = {**header}
	out['FOF'] = {**group_data}
	for pt in ['0', '1', '4']:
		out[f'partType{pt}'] = {**part_data[f'partType{pt}']}
	return out


def glance_cluster(cluster_dict: dict, verbose: bool = False, indent: int = 1) -> None:

	if not verbose:
		for key, value in cluster_dict.items():
			if isinstance(value, dict):
				pprint('\t'*indent + str(key))
				glance_cluster(value, indent=indent+1)
			elif isinstance(value, np.ndarray) and len(value) > 10:
				pprint('\t' * indent + str(key) + ' : ' + f"len({len(value):d})\t val({value[0]} ... {value[-1]})")
			else:
				pprint('\t' * indent + str(key) + ' : ' + str(value))

	if verbose:
		for key, value in cluster_dict.items():
			if isinstance(value, dict):
				pprint('\t'*indent + str(key))
				glance_cluster(value, indent=indent+1)
			else:
				pprint('\t'*indent + str(key) +' : '+ str(value))