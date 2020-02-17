

import numpy as np
import yt
ds = yt.load('/cosma5/data/dp004/C-EAGLE/Complete_Sample/CE_00/data/particledata_029_z000p000/eagle_subfind_particles_029_z000p000.0.hdf5')


# This is an object that describes the entire box
ad = ds.all_data()

# We plot the average velocity magnitude (mass-weighted) in our object
# as a function of density and temperature
plot = yt.PhasePlot(ad, "density", "temperature", "velocity_magnitude")

# save the plot
plot.save('~/rendering.png')
