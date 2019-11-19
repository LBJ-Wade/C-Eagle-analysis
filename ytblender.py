import yt
import numpy as np
import yt.units as units
import pylab
from cluster import Simulation, Cluster

def YT_plot_gas_density(cluster):

    r200 = cluster.group_r200()
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
    density = ad[("PartType0","density")]
    wdens = np.where(density == np.max(density))
    coordinates = ad[("PartType0","Coordinates")]
    center = coordinates[wdens][0]
    print ('center = ',center)
    new_box_size = ds.quan(10*r200,'code_length')

    left_edge = center - new_box_size/2
    right_edge = center + new_box_size/2

    print (new_box_size.in_units('Mpc'))
    print (left_edge.in_units('Mpc'))
    print (right_edge.in_units('Mpc'))
    # ad2= ds.region(center=center, left_edge=left_edge, right_edge=right_edge)
    px = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), center=center, width=new_box_size)
    px.save('cluster_{}'.format(cluster.clusterID))

    # Draw a velocity vector every 16 pixels.
    px = yt.ProjectionPlot(ds, 'x', ('gas', 'temperature'), center=center, width=new_box_size)
    px.annotate_velocity(factor=16)
    px.save('cluster_{}'.format(cluster.clusterID))

    px = yt.ProjectionPlot(ds, 'x', ('gas', 'metallicity'), center=center, width=new_box_size)
    px.save('cluster_{}'.format(cluster.clusterID))


def YT_multislice(cluster):
    r200 = cluster.group_r200()
    fname = cluster.partdata_filePaths()[0]  # dataset to load
    print(fname)

    unit_base = {'UnitLength_in_cm': 3.08568e+21,
                 'UnitMass_in_g': 1.989e+43,
                 'UnitVelocity_in_cm_per_s': 100000}

    bbox_lim = 1e5  # kpc

    bbox = [[-bbox_lim, bbox_lim],
            [-bbox_lim, bbox_lim],
            [-bbox_lim, bbox_lim]]

    ds = yt.load(fname, unit_base=unit_base, bounding_box=bbox)
    ad = ds.all_data()
    density = ad[("PartType0", "density")]
    wdens = np.where(density == np.max(density))
    coordinates = ad[("PartType0", "Coordinates")]
    center = coordinates[wdens][0]

    # Create density slices of several fields along the x axis
    yt.SlicePlot(ds, 'z', [('gas', 'density'), ('gas', 'temperature')],
                 width=(15*r200, 'Mpc'), center=center).save()

cluster = Cluster(clusterID = 4, redshift = 0.101)
YT_plot_gas_density(cluster)