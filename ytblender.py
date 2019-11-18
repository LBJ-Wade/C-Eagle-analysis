import yt
from cluster import Cluster

cluster = Cluster(clusterID = 4, redshift = 0.101)
fn = cluster.partdata_filePaths()[0] # dataset to load

ds = yt.load(fn) # load data
p = yt.SlicePlot(ds, "x", "density")

# Draw a velocity vector every 16 pixels.
p.annotate_velocity(factor = 16)
p.save("%s_3x2" % ds)
