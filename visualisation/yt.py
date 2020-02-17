
def generate_volume():
    import numpy as np
    import yt

    ds = yt.load('/cosma5/data/dp004/C-EAGLE/Complete_Sample/CE_00/data/particledata_029_z000p000/eagle_subfind_particles_029_z000p000.0.hdf5')

    sc = yt.create_scene(ds, lens_type='perspective')

    source = sc[0]

    source.set_field('density')
    source.set_log(True)

    bounds = (3e-31, 5e-27)

    # Since this rendering is done in log space, the transfer function needs
    # to be specified in log space.
    tf = yt.ColorTransferFunction(np.log10(bounds))

    tf.add_layers(5, colormap='arbre')

    source.tfh.tf = tf
    source.tfh.bounds = bounds

    source.tfh.plot('transfer_function.png', profile_field='density')

    sc.save('~/rendering.png', sigma_clip=6)


generate_volume()