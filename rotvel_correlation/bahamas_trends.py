import os, sys
import itertools
import warnings
import numpy as np
import slack
import socket

# Graphics packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
plt.style.use("mnras.mplstyle")

# Internal packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from . import utils

def save_plot(filepath: str, to_slack: bool = False, **kwargs) -> None:
	"""
	Function to parse the plt.savefig method and send the file to a slack channel.
	:param filepath: The path+filename+extension where to save the figure
	:param to_slack: Gives the option to send the the plot to slack
	:param kwargs: Other kwargs to parse into the plt.savefig
	:return: None
	"""
	plt.savefig(filepath, **kwargs)
	if to_slack:
		# Send files to Slack: init slack client with access token
		print(
				"[+] Forwarding to the `#personal` Slack channel",
				f"\tDir: {os.path.dirname(filepath)}",
				f"\tFile: {os.path.basename(filepath)}",
				sep='\n'
		)
		slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
		slack_msg = f"Host: {socket.gethostname()}\nDir: {os.path.dirname(filepath)}\nFile: {os.path.basename(filepath)}"
		try:
			client = slack.WebClient(token=slack_token)
			client.files_upload(file=filepath, initial_comment=slack_msg, channels='#personal')
		except:
			warnings.warn("[-] Failed to broadcast plot to Slack channel.")



redshift = 'z000p000'
aperture = 7
#-----------------------------------------------------------------

m500 = utils.read_snap_output(redshift, apertureID=aperture, dataset='m500')
c_l  = utils.read_snap_output(redshift, apertureID=aperture, dataset='c_l')
figname = f'bahamas_hyd_alignment_{redshift}.png'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("linear")
ax.set_xlabel(utils.datasets_names['m500'])
ax.set_ylabel(utils.datasets_names['c_l'])

ax.scatter(m500, c_l, marker='.', c='yellowgreen', s=3, alpha=0.3, label='load')
save_plot(os.path.join(utils.basepath, figname), to_slack=True, dpi=300)

















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