"""
------------------------------------------------------------------
FILE:   centre_of_mass.py
AUTHOR: Edo Altamura
DATE:   19-11-2019
------------------------------------------------------------------
This file is part of the 'testing' package. Over the last month or
so, the author spotted a problem with the calculation of the centre
of mass (CoM) in C-EAGLE and CELR-eagle data. The calculation of
the merging index resulted to be surprisingly high, due to the centre
of mass being shared between the central halo and come of the
neighbouring ones. The current approach uses the centre of potential
(CoP) as prior for the CoM. The CoM is then calculated after filtering
only the particles which had groupNumber == 1 AND coordinates falling
within the R500 sphere *centred on CoP*.
In order to minimise the bias due to the effect of the CoP as prior,
the sphere needs to be:
    - large enough to avoid bias selection due to the CoP
    - small enough, so that neighbouring halos do not shift the CoM
        outside the main halo's R500 sphere.

This file contains alternative versions of the methods in the cluster
package. These alternative methods are used to validate the ones in
the stable version of the code and provide feedback for more accurate
strategies.
-------------------------------------------------------------------
"""

import numpy as np
from matplotlib import pyplot as plt

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Cluster
from rendering import Map
import map_plot_parameters as plotpar
plotpar.set_defaults_plot()

def group_centre_of_mass(cluster, out_allPartTypes=False, filter_radius = None):
    """
    out_allPartTypes = (bool)
        if True outputs the centre of mass and sum of masses of each
        partType separately in arrays

        if False outputs the overall CoM and sum of masses of the whole
        cluster.

    Returns the centre of mass of the cluster for a ALL particle types,
    except for lowres_DM (2, 3).
    """
    CoM_PartTypes = np.zeros((0, 3), dtype=np.float)
    Mtot_PartTypes = np.zeros(0, dtype=np.float)

    for part_type in ['0', '1', '4', '5']:
        # Import data
        mass = cluster.particle_masses(part_type)
        coords = cluster.particle_coordinates(part_type)
        group_num = cluster.group_number_part(part_type)

        # Filter the particles belonging to the
        # GroupNumber FOF == 1, which by definition is centred in the
        # Centre of Potential and is disconnected from other FoF groups.
        radial_dist = np.linalg.norm(np.subtract(coords, cluster.group_centre_of_potential()), axis=1)
        index = np.where((group_num == 1) & (radial_dist < filter_radius))[0]
        mass = mass[index]
        coords = coords[index]
        assert mass.__len__() > 0, "Array is empty - check filtering."
        assert coords.__len__() > 0, "Array is empty - check filtering."
        # print('Computing CoM ==> PartType {} ok!'.format(part_type))

        # Compute CoM for each particle type
        centre_of_mass, sum_of_masses = cluster.centre_of_mass(mass, coords)
        CoM_PartTypes = np.append(CoM_PartTypes, [centre_of_mass], axis=0)
        Mtot_PartTypes = np.append(Mtot_PartTypes, [sum_of_masses], axis=0)

    if out_allPartTypes:
        return CoM_PartTypes, Mtot_PartTypes
    else:
        return cluster.centre_of_mass(Mtot_PartTypes, CoM_PartTypes)


# Create cluster object
cluster = Cluster(clusterID = 4, redshift = 0.101)

# Define key sizes within the cluster
r500 = cluster.group_r500()
r200 = cluster.group_r200()

# Centre the system on the CoP
CoP = cluster.group_centre_of_potential()
special_markers = [CoP]

# Create array of test values for the sphere-filter
r_filters = np.linspace(0.01*r500, 3*r200, 10)
for r_filter in r_filters:
    CoM, _ = group_centre_of_mass(cluster,
                                  out_allPartTypes = False,
                                  filter_radius = r_filter)
    special_markers.append(CoM)

# Recentre to the CoP and plot
special_markers = np.subtract(special_markers, CoP)
special_markers_labels = ['']*len(special_markers)

CoM_map = Map()
CoM_map.xyz_projections(xyzdata = None,
                          weights = None,
                          plot_limit = 1.2*r200,
                          nbins = None,
                          circle_pars = (0, 0, r500),
                          special_markers = special_markers,
                          special_markers_labels = special_markers_labels,
                          s=0.1)
plt.show()
