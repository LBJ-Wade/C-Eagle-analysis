import yt
import numpy as np
import yt.units as units
import pylab
from cluster import Cluster

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
    # Draw a velocity vector every 16 pixels.
    px.annotate_velocity(factor=16)


    px.save('{}'.format(cluster.clusterID))


def YT_plot_metal_density(cluster):

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

    # Create a derived field, the metal density.
    def _metal_density(field, data):
        density = data['PartType0', 'Density']
        Z = data['PartType0', 'metallicity']
        return density * Z

    # Add it to the dataset.
    ds.add_field(('PartType0', 'metal_density'), function=_metal_density,
                 units="g/cm**3", particle_type=True)

    # Add the corresponding smoothed field to the dataset.
    from yt.fields.particle_fields import add_volume_weighted_smoothed_field

    add_volume_weighted_smoothed_field('PartType0', 'Coordinates', 'Masses',
                                       'SmoothingLength', 'Density',
                                       'metal_density', ds.field_info)

    # Define the region where the disk galaxy is. (See the Gadget notebook for
    # details. Here I make the box a little larger than needed to eliminate the
    # margin effect.)
    center = ds.arr([31996, 31474, 28970], "code_length")
    box_size = ds.quan(10*r200, "code_length")
    left_edge = center - box_size / 2 * 1.1
    right_edge = center + box_size / 2 * 1.1
    box = ds.box(left_edge=left_edge, right_edge=right_edge)

    # And make a projection plot!
    yt.ProjectionPlot(ds, 'z',
                      ('deposit', 'PartType0_smoothed_metal_density'),
                      center=center, width=box_size, data_source=box).save('{}'.format(cluster.clusterID))

def YT_multislice(cluster):


    r200 = cluster.group_r200()
    fname = cluster.partdata_filePaths()[0]  # dataset to load
    print(fname)

    ds = yt.load(fname)

    # Create density slices of several fields along the x axis
    yt.SlicePlot(ds, 'x', ['density', 'temperature', 'pressure'],
                 width=(5*r200, 'Mpc')).save()

cluster = Cluster(clusterID = 4, redshift = 0.101)
YT_multislice(cluster)