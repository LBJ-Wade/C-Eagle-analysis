from clusters_retriever import *
from cluster_profiler import *
import redshift_catalogue_ceagle as zcat

ceagle = Simulation()
z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
cluster = Cluster(clusterID = 0, redshift = 0.101, subject = 'particledata')
print(total_mass_rest_frame(cluster))
