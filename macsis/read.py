import os
from typing import List, Dict
import numpy as np
import h5py as h5
from scipy.sparse import csr_matrix
from mpi4py import MPI

from .__init__ import (
	pprint,
	comm,
	rank,
	nproc,
)
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
    for j in range(nproc):
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

def find_files(redshift: str) -> list:
    z_value = ['z004p688', 'z004p061', 'z003p053', 'z003p078', 'z002p688', 'z002p349',
            'z002p053', 'z001p792', 'z001p561', 'z001p354', 'z001p168', 'z001p000', 'z000p846', 'z000p706',
            'z000p577', 'z000p457', 'z000p345', 'z000p240', 'z000p140', 'z000p046', 'z000p000']
    z_IDNumber = ['002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
            '014', '015', '016', '017', '018', '019', '020', '021', '022']
    sn = dict(zip(z_value, z_IDNumber))[redshift]
    path='/cosma5/data/dp004/dc-hens1/macsis/macsis_gas'
    pprint(f"[+] Find simulation files {redshift:s}...")

    halos = [x for x in os.listdir(path) if x.startswith('halo_')]
	master_files = []
	for halo in halos:
		groups = []
		for x in os.listdir(os.path.join(path, halo, f'data/groups_{sn}')):
			if x.startswith('eagle_subfind_tab'):
				complete_path = os.path.join(path, halo, f'data/groups_{sn}/{x}')
				groups.append(complete_path)

		particle = []
		for x in os.listdir(os.path.join(path, halo, f'data/particledata_{sn}')):
			if x.startswith('eagle_subfind_particles'):
				complete_path = os.path.join(path, halo, f'data/particledata_{sn}/{x}')
				particle.append(complete_path)

		master_files.append([groups[0], particle[0]])

	return master_files



def fof_header(files: list):
	pprint(f"[+] Find header information...")
	header = {}
	with h5.File(files[0][1], 'r') as f:
		header['Hub']  = f['Header'].attrs['HubbleParam']
		header['aexp'] = f['Header'].attrs['ExpansionFactor']
		header['zred'] = f['Header'].attrs['Redshift']
		header['OmgL'] = f['Header'].attrs['OmegaLambda']
		header['OmgM'] = f['Header'].attrs['Omega0']
		header['OmgB'] = f['Header'].attrs['OmegaBaryon']
	return header



def fof_groups(files: list):
	pprint(f"[+] Find groups information...")
	group_files = [pair[0] for pair in files]
	st, fh = split(len(group_files))
	Mfof = np.empty(0, dtype=np.float32)
	M2500 = np.empty(0, dtype=np.float32)
	M500 = np.empty(0, dtype=np.float32)
	M200 = np.empty(0, dtype=np.float32)
	R2500 = np.empty(0, dtype=np.float32)
	R500 = np.empty(0, dtype=np.float32)
	R200 = np.empty(0, dtype=np.float32)
	COP = np.empty(0, dtype=np.float32)
	NSUB = np.empty(0, dtype=np.int)
	FSID = np.empty(0, dtype=np.int)
	SCOP = np.empty(0, dtype=np.float32)
	for x in range(st, fh, 1):
		with h5.File(group_files[x], 'r') as f:
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

	header = {}
	with h5.File(group_files[0], 'r') as f:
		header['Hub'] =  f['Header'].attrs['HubbleParam']
		header['aexp'] = f['Header'].attrs['ExpansionFactor']
		header['zred'] = f['Header'].attrs['Redshift']

	# Conversion
	Mfof = comoving_mass(header, Mfof * 1.0e10)
	M2500 = comoving_mass(header, M2500 * 1.0e10)
	M500 = comoving_mass(header, M500 * 1.0e10)
	M200 = comoving_mass(header, M200 * 1.0e10)
	R2500 = comoving_length(header, R2500)
	R500 = comoving_length(header, R500)
	R200 = comoving_length(header, R200)
	COP = comoving_length(header, COP)
	SCOP = comoving_length(header, SCOP)

	data = {}
	data['groupfiles'] = np.asarray(group_files)
	data['particlefiles'] = np.asarray([pair[1] for pair in files])
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

	return data

def fof_group(clusterID: int, fofgroups: Dict[str, np.ndarray] = None):
	pprint(f"[+] Find group information for cluster {clusterID}")
	new_data = {}
	new_data['clusterID'] = clusterID
	new_data['Mfof']  = fofgroups['Mfof'][clusterID]
	new_data['M2500'] = fofgroups['M2500'][clusterID]
	new_data['R2500'] = fofgroups['R2500'][clusterID]
	new_data['M500']  = fofgroups['M500'][clusterID]
	new_data['R500']  = fofgroups['R500'][clusterID]
	new_data['M200']  = fofgroups['M200'][clusterID]
	new_data['R200']  = fofgroups['R200'][clusterID]
	new_data['COP']   = fofgroups['COP'][clusterID]
	new_data['NSUB']  = fofgroups['NSUB'][clusterID]
	new_data['FSID']  = fofgroups['FSID'][clusterID]
	new_data['SCOP']  = fofgroups['SCOP'][clusterID]
	new_data['groupfiles']  = fofgroups['groupfiles'][clusterID]
	new_data['particlefiles'] = fofgroups['particlefiles'][clusterID]
	return new_data


def cluster_partgroupnumbers(fofgroup: Dict[str, np.ndarray] = None):
	"""

	:param fofgroups:
	:return:
	"""
	pgn = []
	with h5.File(fofgroup['particlefiles'], 'r') as h5file:

		for pt in ['0', '1', '4']:
			Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][int(pt)]
			st, fh = split(Nparticles)
			pprint(f"[+] Collecting particleType {pt} GroupNumber...")
			groupnumber = h5file[f'/PartType{pt}/GroupNumber'][st:fh]

			# Clip out negative values and exceeding values
			groupnumber = np.clip(groupnumber, 0, 6)
			pprint(f"\t Computing CSR indexing matrix...")
			groupnumber_csrm = get_indices_sparse(groupnumber)
			del groupnumber_csrm[0], groupnumber_csrm[-1]
			pgn.append(groupnumber_csrm)
			del groupnumber

	return pgn

def cluster_particles(fofgroup: Dict[str, np.ndarray] = None, groupNumbers: List[np.ndarray] = None):
	"""

	:param fofgroup:
	:param groupNumbers:
	:return:
	"""
	pprint(f"[+] Find particle information for cluster {fofgroup['clusterID']}")
	data_out = {}
	header = {}
	partTypes = ['0', '1', '4']
	with h5.File(fofgroup['particlefiles'], 'r') as h5file:

		header['Hub']  = h5file['Header'].attrs['HubbleParam']
		header['aexp'] = h5file['Header'].attrs['ExpansionFactor']
		header['zred'] = h5file['Header'].attrs['Redshift']

		for pt in partTypes:

			# Initialise particledata arrays
			pgn_core = np.empty(0, dtype=np.int)
			subgroup_number = np.empty(0, dtype=np.int)
			velocity = np.empty(0, dtype=np.float32)
			coordinates = np.empty(0, dtype=np.float32)
			mass = np.empty(0, dtype=np.float32)
			temperature = np.empty(0, dtype=np.float32)
			sphdensity = np.empty(0, dtype=np.float32)
			sphlength = np.empty(0, dtype=np.float32)

			# Let each CPU core import a portion of the pgn data
			pgn = groupNumbers[partTypes.index(pt)]
			st, fh = split(len(pgn))
			pgn_core = np.append(pgn_core, pgn[st:fh])
			del pgn

			# Filter particle data with collected groupNumber indexing
			subgroup_number = np.append(subgroup_number, h5file[f'/PartType{pt}/SubGroupNumber'][pgn_core])
			velocity        = np.append(velocity, h5file[f'/PartType{pt}/Velocity'][pgn_core])
			coordinates     = np.append(coordinates, h5file[f'/PartType{pt}/Coordinates'][pgn_core])
			if pt == '1':
				particle_mass_DM = h5file['Header'].attrs['MassTable'][1]
				mass = np.append(mass, np.ones(len(pgn_core), dtype=np.float32) * particle_mass_DM)
			else:
				mass = np.append(mass, h5file[f'/PartType{pt}/Mass'][pgn_core])
			if pt == '0':
				temperature = np.append(temperature, h5file[f'/PartType{pt}/Temperature'][pgn_core])
				sphdensity  = np.append(sphdensity, h5file[f'/PartType{pt}/Density'][pgn_core])
				sphlength   = np.append(sphlength, h5file[f'/PartType{pt}/SmoothingLength'][pgn_core])

			del pgn_core

			# Conversion from comoving units to physical units
			velocity = comoving_velocity(header, velocity)
			coordinates = comoving_length(header, coordinates)
			mass = comoving_mass(header, mass * 1.0e10)
			if pt == '0':
				den_conv = h5file[f'/PartType{pt}/Density'].attrs['CGSConversionFactor']
				sphdensity = comoving_density(header, sphdensity * den_conv)
				sphlength = comoving_length(header, sphlength)

			# Gather the imports across cores
			data_out[f'partType{pt}'] = {}
			data_out[f'partType{pt}']['subgroupnumber'] = commune(subgroup_number)
			data_out[f'partType{pt}']['velocity']        = commune(velocity.reshape(-1, 1)).reshape(-1, 3)
			data_out[f'partType{pt}']['coordinates']     = commune(coordinates.reshape(-1, 1)).reshape(-1, 3)
			data_out[f'partType{pt}']['mass']            = commune(mass)
			if pt == '0':
				data_out[f'partType{pt}']['temperature'] = commune(temperature)
				data_out[f'partType{pt}']['sphdensity']  = commune(sphdensity)
				data_out[f'partType{pt}']['sphlength']   = commune(sphlength)

			del subgroup_number, velocity, mass, coordinates, temperature, sphdensity, sphlength

	return data_out

def cluster_data(clusterID: int,
                 header: Dict[str, float] = None,
                 fofgroups: Dict[str, np.ndarray] = None):
	"""

	:param clusterID:
	:param header:
	:param fofgroups:
	:param groupNumbers:
	:return:
	"""

	group_data  = fof_group(clusterID, fofgroups = fofgroups)
	halo_partgn = cluster_partgroupnumbers(fofgroup=group_data)
	part_data   = cluster_particles(fofgroup=group_data, groupNumbers= halo_partgn)

	out = {}
	out['Header'] = {**header}
	out['FOF'] = {**group_data}
	for pt in ['0', '1', '4']:
		out[f'partType{pt}'] = {**part_data[f'partType{pt}']}
	return out


def glance_cluster(cluster_dict: dict, verbose: bool = False, indent: int = 1) -> None:
	"""

	:param cluster_dict:
	:param verbose:
	:param indent:
	:return:
	"""
	if not verbose:
		for key, value in cluster_dict.items():
			if isinstance(value, dict):
				pprint('\t'*indent + str(key))
				glance_cluster(value, indent=indent+1)
			elif (isinstance(value, np.ndarray) or isinstance(value, list)) and len(value) > 10:
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