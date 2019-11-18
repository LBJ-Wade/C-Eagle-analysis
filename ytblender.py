import yt
from cluster import Cluster

cluster = Cluster(clusterID = 4, redshift = 0.101)

unit_base = {
    'length': (1.0, 'kpc'),
    'velocity': (1.0, 'km/s'),
    'mass': (1.0, 'Msun')
}

fn = cluster.partdata_filePaths()[1] # dataset to load

ds = yt.load(fn, unit_base=unit_base) # load data
p = yt.SlicePlot(ds, "x", "density")

# Draw a velocity vector every 16 pixels.
p.annotate_velocity(factor = 16)
p.save("%s_3x2" % ds)
