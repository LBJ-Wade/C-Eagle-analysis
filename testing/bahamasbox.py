import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import slack
import warnings
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import EllipseCollection

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
filename = f"bahamas-box-5r200spheres.jpg"

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
# ax.set_xlim([0, 400])
# ax.set_ylim([0, 400])
ax.set_xlabel(r'$x$ \quad [cMpc]')
ax.set_ylabel(r'$y$ \quad [cMpc]')

n_largeM = 0
n_total = 0

for counter, file in enumerate(cluster.groups_filePaths()):
	print(f"[+] Analysing eagle_subfind_tab file {counter}")
	with h5py.File(file, 'r') as group_file:
		cop = group_file['/FOF/GroupCentreOfPotential'][:]
		m500 = group_file['/FOF/Group_M_Crit500'][:]*10**10
		r200 = group_file['/FOF/Group_R_Crit200'][:]

		m_filter = np.where(m500 > 10**13)[0]
		ax.scatter(cop[m_filter,0], cop[m_filter,1], marker='o', s=2, c='r', alpha=1)
		offsets = list(zip(cop[m_filter,0], cop[m_filter,1]))
		ax.add_collection(EllipseCollection(widths=r200[m_filter]*5, heights=r200[m_filter]*5, angles=0, units='xy',
		                                    facecolors='r', offsets=offsets, alpha=0.3,
		                                    transOffset=ax.transData))

		m_filter = np.where(m500 < 10 ** 13)[0]
		ax.scatter(cop[m_filter, 0], cop[m_filter, 1], marker='o', s=2, c='k', alpha=1)
		offsets = list(zip(cop[m_filter, 0], cop[m_filter, 1]))
		ax.add_collection(EllipseCollection(widths=r200[m_filter] * 5, heights=r200[m_filter] * 5, angles=0, units='xy',
		                                    facecolors='k', offsets=offsets, alpha=0.3,
		                                    transOffset=ax.transData))

		n_largeM += len(m_filter)
		n_total += len(m500)

print('n_largeM', n_largeM)
print('n_total', n_total)


blue_star = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
                          markersize=10, label=r'$M_{500~crit} < 10^{13}\ M_\odot$')
red_square = mlines.Line2D([], [], color='r', marker='o', linestyle='None',
                          markersize=10, label=r'$M_{500~crit} > 10^{13}\ M_\odot$')
plt.legend(handles=[blue_star, red_square])

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