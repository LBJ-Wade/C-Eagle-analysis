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

def fof_header(files):
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

def fof_groups(files, header: Dict[str, float]):
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

			# Conversion
			Mfof = comoving_mass(header, mass_units(Mfof, unit_system='astro'))
			M2500 = comoving_mass(header, mass_units(M2500, unit_system='astro'))
			R2500 = comoving_length(header, length_units(R2500, unit_system='astro'))
			M500 = comoving_mass(header, mass_units(M500, unit_system='astro'))
			R500 = comoving_length(header, length_units(R500, unit_system='astro'))
			M200 = comoving_mass(header, mass_units(M200, unit_system='astro'))
			R200 = comoving_length(header, length_units(R200, unit_system='astro'))
			COP = comoving_length(header, length_units(COP, unit_system='astro'))
			SCOP = comoving_length(header, length_units(SCOP, unit_system='astro'))

	data = {}
	data['Mfof'] = commune(Mfof)
	data['M2500'] = commune(M2500)
	data['R2500'] = commune(R2500)
	data['M500'] = commune(M500)
	data['R500'] = commune(R500)
	data['M200'] = commune(M200)
	data['R200'] = commune(R200)
	data['COP'] = commune(COP.reshape(-1, 1)).reshape(-1, 3)
	data['NSUB'] = commune(NSUB)
	data['FSID'] = commune(FSID)
	data['SCOP'] = commune(SCOP.reshape(-1, 1)).reshape(-1, 3)
	data['idx'] = np.where(M500 > 1.0e13)[0]
	return data

def snap_groupnumbers(files, fofgroups: Dict[str, np.ndarray]):
	halo_num_catalogue_contiguous = np.max(fofgroups['idx'])

	with h5.File(files[1], 'r') as h5file:
		Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0, 1, 4]]

		pprint(f"[+] Collecting gas particles GroupNumber...")
		st, fh = split(Nparticles[0])
		groupnumber0 = h5file[f'/PartType0/GroupNumber'][st:fh]
		# Clip out negative values and exceeding values
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber0 = np.clip(groupnumber0, 0, np.max(halo_num_catalogue_contiguous) + 2)
		groupnumber0_csrm = get_indices_sparse(groupnumber0)
		del groupnumber0

		pprint(f"[+] Collecting CDM particles GroupNumber...")
		st, fh = split(Nparticles[1])
		groupnumber1 = h5file[f'/PartType1/GroupNumber'][st:fh]
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber1 = np.clip(groupnumber1, 0, np.max(halo_num_catalogue_contiguous) + 2)
		groupnumber1_csrm = get_indices_sparse(groupnumber1)
		del groupnumber1

		pprint(f"[+] Collecting star particles GroupNumber...")
		st, fh = split(Nparticles[2])
		groupnumber4 = h5file[f'/PartType4/GroupNumber'][st:fh]
		pprint(f"[+] Computing CSR indexing matrix...")
		groupnumber4 = np.clip(groupnumber4, 0, np.max(halo_num_catalogue_contiguous) + 2)
		groupnumber4_csrm = get_indices_sparse(groupnumber4)
		del groupnumber4

	return [groupnumber0_csrm, groupnumber1_csrm, groupnumber4_csrm]


def cluster_particles(files, groupNumbers: List[np.ndarray,np.ndarray,np.ndarray] = None):
	with h5.File(files[1], 'r') as h5file:
		data_out = {}
		partTypes = ['0', '1', '4']
		for pt in partTypes:
			groupNumber = groupNumbers[partTypes.index(pt)]
			data_out[f'partType{pt}'] = {}
			data_out[f'partType{pt}']['subgroup_number'] = h5file[f'/PartType{pt}/SubGroupNumber'][groupNumber]
			data_out[f'partType{pt}']['velocity'] = h5file[f'/PartType{pt}/Velocity'][groupNumber]
			data_out[f'partType{pt}']['coordinates'] = h5file[f'/PartType{pt}/SubGroupNumber'][groupNumber]
			if pt == '1':
				particle_mass_DM = h5file['Header'].attrs['MassTable'][1]
				data_out[f'partType{pt}']['mass'] = np.ones(len(groupNumber), dtype=np.float) * particle_mass_DM
			else:
				data_out[f'partType{pt}']['mass'] = h5file[f'/PartType{pt}/Mass'][groupNumber]
			if pt == '0':
				data_out[f'partType{pt}']['temperature'] = h5file[f'/PartType{pt}/Temperature'][groupNumber]
				data_out[f'partType{pt}']['sphdensity'] = h5file[f'/PartType{pt}/Density'][groupNumber]
				data_out[f'partType{pt}']['sphlength'] = h5file[f'/PartType{pt}/SmoothingLength'][groupNumber]

		boxsize = h5file['Header'].attrs['BoxSize']


# def cluster_data(halo_ID):
# 	data = {}
#
# 	# Initialise the allocation for cluster reports
# 	clusterID_pool = np.arange(N_HALOS)
# 	comm.Barrier()
# 	for i in clusterID_pool:
# 		pprint(f"[+] Initializing partGN generation... {SIMULATION:>10s} {i:<5d} {REDSHIFT:s}")
# 		fof_id = halo_num_catalogue_contiguous[i] + 1
# 		pgn0 = commune(groupnumber0_csrm[fof_id][0])
# 		pgn1 = commune(groupnumber1_csrm[fof_id][0])
# 		pgn4 = commune(groupnumber4_csrm[fof_id][0])
# 		pprint(f"\tInitializing report generation...")
# 		alignment.save_report(i, REDSHIFT, glob=[pgn0, pgn1, pgn4])
# 	comm.Barrier()
