import sys
import os
import warnings
import itertools
import subprocess
import numpy as np
import pandas as pd
import h5py
import slack
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


redshift = 'z000p000'
aperture = 7

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

basepath = '/local/scratch/altamura/analysis_results/alignment_project'
snapname = f'bahamas_hyd_alignment_{redshift}.hdf5'
figname = f'bahamas_hyd_alignment_{redshift}.png'

c_l = []
m500 = []

with h5py.File(os.path.join(basepath, snapname), 'r') as file:
	for i in range(30000):
		if f'halo_{i:05d}' in file:
			print(file[f'halo_{i:05d}/aperture{aperture:02d}/c_l'][1,1])
			c_l.append(file[f'halo_{i:05d}/aperture{aperture:02d}/c_l'][1,1])
			m500.append(file[f'halo_{i:05d}/aperture{aperture:02d}/m500'][()])
		else:
			print(f'Halo {i} not found')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("linear")

ax.scatter(m500, c_l, marker='.', c='yellowgreen', s=3, alpha=0.3, label='load')

ax.set_xlabel('M_{500}')
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



