import yt
import numpy as np
import yt.units as units
import pylab
from cluster import Cluster

cluster = Cluster(clusterID = 4, redshift = 0.101)
fname = cluster.partdata_filePaths()[0] # dataset to load
print(fname)

unit_base = {'UnitLength_in_cm': 3.08568e+21,
             'UnitMass_in_g': 1.989e+43,
             'UnitVelocity_in_cm_per_s': 100000}

bbox_lim = 1e5  # kpc

bbox = [[-bbox_lim, bbox_lim],
        [-bbox_lim, bbox_lim],
        [-bbox_lim, bbox_lim]]

ds = yt.load(fname, unit_base=unit_base, bounding_box=bbox)
ds.index
ad = ds.all_data()
px = yt.ProjectionPlot(ds, 'x', ('gas', 'density'))
px.show()
