"""
------------------------------------------------------------------
FILE:   masking.py
AUTHOR: Edo Altamura
DATE:   25-03-2020
------------------------------------------------------------------
This file is part of the 'import_toolkit' package. It contains a
Mask class with methods to select only the particles used in the
analysis, excluding all those which would be thrown out at some
point of the pipeline.
An example of these are the unbound particles with group_number < 0.
The equation of state and particles beyond the high-resolution zoom
regions are other examples of particles which would not be used in
the pipeline.
By cutting out the unwanted particles at runtime, the pipeline is
left with fewer particles to store in the RAM memory, preventing
overflow errors and allowing to use a lower number of cores to achieve
similar speedups.
-------------------------------------------------------------------
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from cluster import Cluster

class Mask(Cluster):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.particle_type = None
        self.condition_container = []

    def set_particle_type(self, particle_type):
        self.particle_type = particle_type

    def filter_group_number(self):
        return np.where(self.group_number_part(self.particle_type) > 0)[0]

    def filter_clean_radius(self):
        coords = self.particle_coordinates(self.particle_type)
        radial_dist = np.linalg.norm(np.subtract(coords, self.group_centre_of_potential()), axis=1)









