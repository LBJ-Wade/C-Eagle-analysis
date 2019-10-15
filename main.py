from clusters_retriever import *

ceagle = Simulation()
z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
cluster1 = Cluster(clusterID = 0, redshift = z_catalogue[-1])
files = cluster1.file_CompletePath_hdf5()
print(files)
