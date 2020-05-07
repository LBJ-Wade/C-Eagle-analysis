import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import slack
import warnings
import h5py
import numpy as np
from matplotlib import pyplot as plt

# exec(open(os.path.abspath(os.path.join(
# 		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
warnings.filterwarnings("ignore")
from import_toolkit.cluster import Cluster

cluster = Cluster(simulation_name='bahamas',
                  clusterID=0,
                  redshift='z000p000',
                  comovingframe=False,
                  fastbrowsing=True)

filepath = "/local/scratch/altamura/analysis_results/"
filename = f"bahamas-box-5r200spheres.pdf"

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
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
		ax.scatter(cop[~m_filter, 0], cop[~m_filter, 1], marker='o', s=5, c='k', alpha=0.1, label=r'$M_{500~crit} < 10^{14}\ M_\odot$')
		ax.scatter(cop[m_filter,0], cop[m_filter,1], marker='o', s=5, c='r', alpha=1, label=r'$M_{500~crit} > 10^{14}\ M_\odot$')


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