# The BAHAMAS simulation is run in the form of a periodic volume and therefore
# requires a separate backend, incompatble with zooms.

from .cluster import Cluster
if __name__ == '__main__':

    import inspect

    class TEST:
        data_required = {'partType0': ['mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
                         'partType1': ['mass', 'coordinates', 'velocity'],
                         'partType4': ['mass', 'coordinates', 'velocity']}

        def cluster_imports(self):
            print(inspect.stack()[0][3])
            cluster = Cluster(simulation_name='bahamas',
                              clusterID=0,
                              redshift='z000p000',
                              comovingframe=False)

            cluster.info()

    test = TEST()
    test.cluster_imports()







"""
Particledata
FILE_CONTENTS {
 group      /
 group      /Config
 group      /Constants
 group      /Header
 group      /Parameters
 group      /Parameters/ChemicalElements
 group      /PartType0
 dataset    /PartType0/Coordinates
 dataset    /PartType0/Density
 group      /PartType0/ElementAbundance
 dataset    /PartType0/ElementAbundance/Carbon
 dataset    /PartType0/ElementAbundance/Helium
 dataset    /PartType0/ElementAbundance/Hydrogen
 dataset    /PartType0/ElementAbundance/Iron
 dataset    /PartType0/ElementAbundance/Magnesium
 dataset    /PartType0/ElementAbundance/Neon
 dataset    /PartType0/ElementAbundance/Nitrogen
 dataset    /PartType0/ElementAbundance/Oxygen
 dataset    /PartType0/ElementAbundance/Silicon
 dataset    /PartType0/GroupNumber
 dataset    /PartType0/HostHalo_TVir_Mass
 dataset    /PartType0/InternalEnergy
 dataset    /PartType0/IronFromSNIa
 dataset    /PartType0/Mass
 dataset    /PartType0/Metallicity
 dataset    /PartType0/OnEquationOfState
 dataset    /PartType0/ParticleIDs
 group      /PartType0/SmoothedElementAbundance
 dataset    /PartType0/SmoothedElementAbundance/Carbon
 dataset    /PartType0/SmoothedElementAbundance/Helium
 dataset    /PartType0/SmoothedElementAbundance/Hydrogen
 dataset    /PartType0/SmoothedElementAbundance/Iron
 dataset    /PartType0/SmoothedElementAbundance/Magnesium
 dataset    /PartType0/SmoothedElementAbundance/Neon
 dataset    /PartType0/SmoothedElementAbundance/Nitrogen
 dataset    /PartType0/SmoothedElementAbundance/Oxygen
 dataset    /PartType0/SmoothedElementAbundance/Silicon
 dataset    /PartType0/SmoothedIronFromSNIa
 dataset    /PartType0/SmoothedMetallicity
 dataset    /PartType0/SmoothingLength
 dataset    /PartType0/StarFormationRate
 dataset    /PartType0/SubGroupNumber
 dataset    /PartType0/Temperature
 dataset    /PartType0/Velocity
 group      /PartType1
 dataset    /PartType1/Coordinates
 dataset    /PartType1/GroupNumber
 dataset    /PartType1/ParticleIDs
 dataset    /PartType1/SubGroupNumber
 dataset    /PartType1/Velocity
 group      /PartType4
 dataset    /PartType4/Coordinates
 dataset    /PartType4/Density
 group      /PartType4/ElementAbundance
 dataset    /PartType4/ElementAbundance/Carbon
 dataset    /PartType4/ElementAbundance/Helium
 dataset    /PartType4/ElementAbundance/Hydrogen
 dataset    /PartType4/ElementAbundance/Iron
 dataset    /PartType4/ElementAbundance/Magnesium
 dataset    /PartType4/ElementAbundance/Neon
 dataset    /PartType4/ElementAbundance/Nitrogen
 dataset    /PartType4/ElementAbundance/Oxygen
 dataset    /PartType4/ElementAbundance/Silicon
 dataset    /PartType4/GroupNumber
 dataset    /PartType4/HostHalo_TVir
 dataset    /PartType4/HostHalo_TVir_Mass
 dataset    /PartType4/InitialMass
 dataset    /PartType4/IronFromSNIa
 dataset    /PartType4/Mass
 dataset    /PartType4/Metallicity
 dataset    /PartType4/ParticleIDs
 group      /PartType4/SmoothedElementAbundance
 dataset    /PartType4/SmoothedElementAbundance/Carbon
 dataset    /PartType4/SmoothedElementAbundance/Helium
 dataset    /PartType4/SmoothedElementAbundance/Hydrogen
 dataset    /PartType4/SmoothedElementAbundance/Iron
 dataset    /PartType4/SmoothedElementAbundance/Magnesium
 dataset    /PartType4/SmoothedElementAbundance/Neon
 dataset    /PartType4/SmoothedElementAbundance/Nitrogen
 dataset    /PartType4/SmoothedElementAbundance/Oxygen
 dataset    /PartType4/SmoothedElementAbundance/Silicon
 dataset    /PartType4/SmoothedIronFromSNIa
 dataset    /PartType4/SmoothedMetallicity
 dataset    /PartType4/SmoothingLength
 dataset    /PartType4/StellarFormationTime
 dataset    /PartType4/SubGroupNumber
 dataset    /PartType4/Velocity
 group      /PartType5
 dataset    /PartType5/Coordinates
 dataset    /PartType5/GroupNumber
 dataset    /PartType5/Mass
 dataset    /PartType5/ParticleIDs
 dataset    /PartType5/SubGroupNumber
 dataset    /PartType5/Velocity
 group      /RuntimePars
 group      /Units
"""

"""
HDF5 "group_tab_031.99.hdf5" {
FILE_CONTENTS {
 group      /
 group      /Constants
 group      /FOF
 dataset    /FOF/CentreOfMass
 dataset    /FOF/GroupLength
 dataset    /FOF/GroupLengthType
 dataset    /FOF/GroupMassType
 dataset    /FOF/GroupOffset
 dataset    /FOF/GroupOffsetType
 dataset    /FOF/Mass
 group      /FOF/NSF
 group      /FOF/NSF/ElementAbundance
 dataset    /FOF/NSF/ElementAbundance/Carbon
 dataset    /FOF/NSF/ElementAbundance/Helium
 dataset    /FOF/NSF/ElementAbundance/Hydrogen
 dataset    /FOF/NSF/ElementAbundance/Iron
 dataset    /FOF/NSF/ElementAbundance/Magnesium
 dataset    /FOF/NSF/ElementAbundance/Neon
 dataset    /FOF/NSF/ElementAbundance/Nitrogen
 dataset    /FOF/NSF/ElementAbundance/Oxygen
 dataset    /FOF/NSF/ElementAbundance/Silicon
 dataset    /FOF/NSF/Entropy
 dataset    /FOF/NSF/Mass
 dataset    /FOF/NSF/Metallicity
 group      /FOF/NSF/SmoothedElementAbundance
 dataset    /FOF/NSF/SmoothedElementAbundance/Carbon
 dataset    /FOF/NSF/SmoothedElementAbundance/Helium
 dataset    /FOF/NSF/SmoothedElementAbundance/Hydrogen
 dataset    /FOF/NSF/SmoothedElementAbundance/Iron
 dataset    /FOF/NSF/SmoothedElementAbundance/Magnesium
 dataset    /FOF/NSF/SmoothedElementAbundance/Neon
 dataset    /FOF/NSF/SmoothedElementAbundance/Nitrogen
 dataset    /FOF/NSF/SmoothedElementAbundance/Oxygen
 dataset    /FOF/NSF/SmoothedElementAbundance/Silicon
 dataset    /FOF/NSF/SmoothedIronFromSNIa
 dataset    /FOF/NSF/SmoothedMetallicity
 dataset    /FOF/NSF/Temperature
 dataset    /FOF/ParticleIDs
 group      /FOF/SF
 group      /FOF/SF/ElementAbundance
 dataset    /FOF/SF/ElementAbundance/Carbon
 dataset    /FOF/SF/ElementAbundance/Helium
 dataset    /FOF/SF/ElementAbundance/Hydrogen
 dataset    /FOF/SF/ElementAbundance/Iron
 dataset    /FOF/SF/ElementAbundance/Magnesium
 dataset    /FOF/SF/ElementAbundance/Neon
 dataset    /FOF/SF/ElementAbundance/Nitrogen
 dataset    /FOF/SF/ElementAbundance/Oxygen
 dataset    /FOF/SF/ElementAbundance/Silicon
 dataset    /FOF/SF/Entropy
 dataset    /FOF/SF/IronFromSNIa
 dataset    /FOF/SF/Mass
 dataset    /FOF/SF/Metallicity
 group      /FOF/SF/SmoothedElementAbundance
 dataset    /FOF/SF/SmoothedElementAbundance/Carbon
 dataset    /FOF/SF/SmoothedElementAbundance/Helium
 dataset    /FOF/SF/SmoothedElementAbundance/Hydrogen
 dataset    /FOF/SF/SmoothedElementAbundance/Iron
 dataset    /FOF/SF/SmoothedElementAbundance/Magnesium
 dataset    /FOF/SF/SmoothedElementAbundance/Neon
 dataset    /FOF/SF/SmoothedElementAbundance/Nitrogen
 dataset    /FOF/SF/SmoothedElementAbundance/Oxygen
 dataset    /FOF/SF/SmoothedElementAbundance/Silicon
 dataset    /FOF/SF/SmoothedIronFromSNIa
 dataset    /FOF/SF/SmoothedMetallicity
 dataset    /FOF/SF/Temperature
 dataset    /FOF/StarFormationRate
 group      /FOF/Stars
 group      /FOF/Stars/ElementAbundance
 dataset    /FOF/Stars/ElementAbundance/Carbon
 dataset    /FOF/Stars/ElementAbundance/Helium
 dataset    /FOF/Stars/ElementAbundance/Hydrogen
 dataset    /FOF/Stars/ElementAbundance/Iron
 dataset    /FOF/Stars/ElementAbundance/Magnesium
 dataset    /FOF/Stars/ElementAbundance/Neon
 dataset    /FOF/Stars/ElementAbundance/Nitrogen
 dataset    /FOF/Stars/ElementAbundance/Oxygen
 dataset    /FOF/Stars/ElementAbundance/Silicon
 dataset    /FOF/Stars/InitialMass
 dataset    /FOF/Stars/InitialMassWeightedStellarAge
 dataset    /FOF/Stars/IronFromSNIa
 dataset    /FOF/Stars/Mass
 dataset    /FOF/Stars/Metallicity
 group      /FOF/Stars/SmoothedElementAbundance
 dataset    /FOF/Stars/SmoothedElementAbundance/Carbon
 dataset    /FOF/Stars/SmoothedElementAbundance/Helium
 dataset    /FOF/Stars/SmoothedElementAbundance/Hydrogen
 dataset    /FOF/Stars/SmoothedElementAbundance/Iron
 dataset    /FOF/Stars/SmoothedElementAbundance/Magnesium
 dataset    /FOF/Stars/SmoothedElementAbundance/Neon
 dataset    /FOF/Stars/SmoothedElementAbundance/Nitrogen
 dataset    /FOF/Stars/SmoothedElementAbundance/Oxygen
 dataset    /FOF/Stars/SmoothedElementAbundance/Silicon
 dataset    /FOF/Stars/SmoothedIronFromSNIa
 dataset    /FOF/Stars/SmoothedMetallicity
 dataset    /FOF/Velocity
 group      /Header
 group      /Parameters
 group      /Parameters/ChemicalElements
 group      /Units
 """
