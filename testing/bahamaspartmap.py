import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import slack
import warnings
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

import matplotlib.lines as mlines
from matplotlib.collections import EllipseCollection

# exec(open(os.path.abspath(os.path.join(
# 		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
warnings.filterwarnings("ignore")
from import_toolkit.cluster import Cluster

data_required = {'partType0': ['groupnumber', 'mass', 'coordinates']}

cluster = Cluster(simulation_name='bahamas',
                  clusterID=0,
                  redshift='z003p000',
                  comovingframe=False,
                  fastbrowsing=False,
                  requires=data_required)

filepath = "/local/scratch/altamura/analysis_results/"
filename = f"bahamas-clustermap-5r200.jpg"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel(r'$y\ $ [Mpc]')
ax.set_ylabel(r'$y\ $ [Mpc]')

coords = getattr(cluster, f'partType0_coordinates')
x0, y0, z0 = cluster.centre_of_potential
a, b, c = cluster.principal_axes_ellipsoid(cluster.inertia_tensor(cluster.partType0_mass, coords))
x = coords[:,0]
y = coords[:,1]
del coords
ax.scatter(x,y, marker=',', c='k', s=1, alpha=0.01)
ax.scatter([x0], [y0], marker='*', c='r', s=10, alpha=1)

# e1 = patches.Ellipse((x0, y0), a*cluster.r200, b*cluster.r200,
#                      angle=cluster.angle_between_vectors(a, [1,0,0]),
#                      linewidth=2, fill=False, color='r')
#
# ax.add_patch(e1)

items_labels = r"""POINT PARTICLE MAP
Cluster {:s} {:d}
$z$ = {:.2f}
$R_{{500\ true}}$ = {:.2f} Mpc""".format(cluster.simulation,
                                          cluster.clusterID,
                                          cluster.z,
                                          cluster.r500)

print(items_labels)
ax.text(0.03, 0.97, items_labels,
          horizontalalignment='left',
          verticalalignment='top',
          transform=ax.transAxes,
          size=15)

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