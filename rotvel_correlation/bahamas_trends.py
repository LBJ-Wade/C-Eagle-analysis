import os
import itertools
import warnings
import numpy as np
import pandas as pd
import h5py
from typing import Dict
from multiprocessing.dummy import Pool as ThreadPool
import slack
import scipy.stats as st

# Graphics packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("mnras.mplstyle")

# Internal packages
from .utils import datasets_names, aperture_labels


redshift = 'z000p000'
aperture = 7
#-----------------------------------------------------------------

basepath = '/local/scratch/altamura/analysis_results/alignment_project'

def pull_halo_output(h5file, clusterID, apertureID, dataset):
	"""
	Function to extract a dataset from a Bahamas snapshot output.
	:param h5file: The h5py File object to extract the data from
	:param clusterID: The number of cluster in the order imposed by FoF
	:param apertureID: int(0-22) The index of the spherical aperture centred on the CoP
	:param dataset: The name of the dataset to pull
	:return: None if the cluster does not exist in the file, the dataset as np.ndarray if it exists
	"""
	if f'halo_{clusterID:05d}' not in h5file:
		warnings.warn(f"[-] Cluster {clusterID} not found in snap output.")
		return None
	else:
		return h5file[f'halo_{clusterID:05d}/aperture{apertureID:02d}/{dataset}'][...]

def read_snap_output(redshift: str, apertureID: int = 7, dataset: str = None) -> np.ndarray:
	"""
	Function to collect datasets from all clusters at given redshift and aperture.
	The function is a wrapper around `pull_halo_output`, called within the multiprocessing/threading
	modules. It multithreads the I/O function without the use of MPI for performance.
	:param redshift: The redshift as a string in EAGLE format
	:param apertureID: int(0-22) The index of the spherical aperture centred on the CoP
	:param dataset: The name of the dataset to pull
	:return: The merged dataset selected taken from all snapshot clusters.
	"""
	snapname = f'bahamas_hyd_alignment_{redshift}.hdf5'
	h5file = h5py.File(os.path.join(basepath, snapname), 'r')
	last_halo_id = int(h5file.keys()[:-1][-5:])
	clusterIDs = list(range(last_halo_id))

	# Make the Pool of workers
	pool = ThreadPool(12)
	results = pool.starmap(
			pull_halo_output,
			zip(
					itertools.repeat(h5file),
					clusterIDs,
					itertools.repeat(apertureID),
					itertools.repeat(dataset)
			)
	)

	# Close the pool and wait for the work to finish
	h5file.close()
	pool.close()
	pool.join()
	results = [x for x in results if x != None]
	results = np.asarray(results)
	return results

def output_as_dict(redshift: str, apertureID: int = 7) -> Dict[str, np.ndarray]:
	snap_out = {}
	for dataset in datasets_names:
		snap_out[dataset] = read_snap_output(redshift, apertureID=apertureID, dataset=dataset)
	return snap_out

def output_as_pandas(redshift: str, apertureID: int = 7) -> pd.DataFrame:
	snap_dict = output_as_pandas(redshift, apertureID = apertureID)
	snap_pd = pd.DataFrame(data=snap_dict, columns=snap_dict.keys())
	del snap_dict
	return snap_pd

m500 = read_snap_output(redshift, apertureID=aperture, dataset='m500')
c_l  = read_snap_output(redshift, apertureID=aperture, dataset='c_l')
figname = f'bahamas_hyd_alignment_{redshift}.png'


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("linear")

ax.scatter(m500, c_l, marker='.', c='yellowgreen', s=3, alpha=0.3, label='load')

ax.set_xlabel('$M_{500}$')
ax.set_ylabel('C - L')
plt.savefig(os.path.join(basepath, figname), dpi=300)













# Send files to Slack: init slack client with access token
print(f"[+] Forwarding {redshift} plot to the `#personal` Slack channel...")
slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
client = slack.WebClient(token=slack_token)
response = client.files_upload(
		file=os.path.join(basepath, figname),
		initial_comment=f"This file was sent upon completion of the plot factory pipeline.\nAttachments: {figname}",
		channels='#personal'
)



"""
APERTURES

physical = [0.03, 0.05, 0.07, 0.10]
manual = [
        0.1*self.r500,
        self.r2500,
        1.5*self.r2500,
        self.r500
]
auto = np.logspace(np.log10(self.r200), np.log10(5 * self.r200), 15).tolist()
all_apertures = physical+manual+auto

DATASETS (f['halo_00000/aperture00'])

N_particles
NumOfSubhalos
Omega0
OmegaBaryon
OmegaLambda
a_l
a_v
a_w
angular_momentum
angular_velocity
aperture_mass
b_l
b_v
b_w
c_l
c_v
c_w
centre_of_mass
centre_of_potential
circular_velocity
dynamical_merging_index
eigenvalues
eigenvectors
elongation
hubble_param
inertia_tensor
kinetic_energy
l_w
m200
m2500
m500
mfof
r200
r2500
r500
r_aperture
redshift
specific_angular_momentum
sphericity
spin_parameter
substructure_fraction
substructure_mass
thermal_energy
thermodynamic_merging_index
triaxiality
v_l
v_w
zero_momentum_frame
"""