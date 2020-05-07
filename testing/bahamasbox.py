import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import slack
import warnings
import h5py

import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


exec(open(os.path.abspath(os.path.join(
		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rotvel_correlation.simstats import Simstats
warnings.filterwarnings("ignore")
from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from visualisation import rendering
from import_toolkit._cluster_retriever import redshift_str2num

import inspect

data_required = {'partType0': ['mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
                 'partType1': ['mass', 'coordinates', 'velocity'],
                 'partType4': ['mass', 'coordinates', 'velocity']}

print(inspect.stack()[0][3])
cluster = Cluster(simulation_name='bahamas',
                  clusterID=0,
                  redshift='z000p000',
                  comovingframe=False)

filepath = "/local/scratch/altamura/analysis_results/"
filename = f"bahamas-box-5r200spheres.pdf"

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot()
ax.set_aspect('equal')
ax.set_xlim([0, 400])
ax.set_ylim([0, 400])
ax.set_xlabel(r'$x$ \quad [cMpc]')
ax.set_ylabel(r'$y$ \quad [cMpc]')

for counter, file in enumerate(cluster.groups_filePaths()):
	print(f"[+] Analysing eagle_subfind_tab file {counter}")
	with h5py.File(file, 'r') as group_file:
		cop = group_file['/FOF/GroupCentreOfPotential'][:]
		m500 = group_file['/FOF/Group_M_Crit500'][:]*10**10
		m_filter = np.where(m500>10**14)[0]
		ax.scatterplot(~cop[m_filter, 0], cop[~m_filter, 1], marker='o', size=5, color='k', alpha=0.1, label=r'$M_{500~crit} < 10^{14}\ M_\odot$')
		ax.scatterplot(cop[m_filter,0], cop[m_filter,1], marker='o', size=5, color='r', alpha=1, label=r'$M_{500~crit} > 10^{14}\ M_\odot$')


legend = ax.legend(loc='upper center', shadow=True)
legend.get_frame().set_facecolor('white')
plt.savefig(filepath+filename)

# Send files to Slack: init slack client with access token
print(f"[+] Forwarding {filename} to the `#personal` Slack channel...")
slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
client = slack.WebClient(token=slack_token)
response = client.files_upload(
        file=f"{filepath+filename}",
        initial_comment=f"This file was sent upon completion of the plot factory pipeline.\nAttachments: {filename}",
        channels='#personal'
)