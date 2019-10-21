from clusters_retriever import *

ceagle = Simulation()
z_catalogue = ceagle.get_redshiftAllowed(dtype = float)
cluster1 = Cluster(clusterID = 0, redshift = z_catalogue[-1],
                    subject = 'groups')

print(cluster1.group_centre_of_potential())
print(cluster1.group_centre_of_mass())

print(cluster1.group_r200())
# print(cluster1.Nsubgroups)
