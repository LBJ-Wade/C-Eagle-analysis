"""
------------------------------------------------------------------
FILE:   angular momentum.py
AUTHOR: Edo Altamura
DATE:   19-11-2019
------------------------------------------------------------------
This file is part of the 'testing' package.
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


def angular_momentum_PartType_alignment_matrix(cluster):
    ang_momenta, sum_of_masses = cluster.group_angular_momentum(out_allPartTypes=True)


cluster = Cluster(clusterID=4, redshift=0.101)
angular_momentum_PartType_alignment_matrix(cluster)