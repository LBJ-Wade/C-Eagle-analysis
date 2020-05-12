# The BAHAMAS simulation is run in the form of a periodic volume and therefore
# requires a separate backend, incompatble with zooms.



#!/usr/bin/env python
import os
import h5py
import numpy as np
from mpi4py import MPI
np.seterr(divide='ignore')

# ----- REQUIRED FUNCTIONS -----
def split(NProcs,MyRank,nfiles):
    nfiles=int(nfiles)
    nf=int(nfiles/NProcs)
    rmd=nfiles % NProcs
    st=MyRank*nf
    fh=(MyRank+1)*nf
    if MyRank < rmd:
        st+=MyRank
        fh+=(MyRank+1)
    else:
        st+=rmd
        fh+=rmd
    return st,fh

def commune(comm,MyRank,NProcs,data):
    tmp=np.zeros(NProcs,dtype=np.int)
    tmp[MyRank]=len(data)
    cnts=np.zeros(NProcs,dtype=np.int)
    comm.Allreduce([tmp,MPI.INT],[cnts,MPI.INT],op=MPI.SUM)
    del tmp
    dspl=np.zeros(NProcs,dtype=np.int)
    i=0
    for j in range(0,NProcs,1):
        dspl[j]=i
        i+=cnts[j]
    rslt=np.zeros(i,dtype=data.dtype)
    comm.Allgatherv([data,cnts[MyRank]],[rslt,cnts,dspl,MPI._typedict[data.dtype.char]])
    del data,cnts,dspl
    return rslt

def slice_conv(f,iden,hub,aexp,st,fh):
    ase=aexp**(f[iden].attrs['aexp-scale-exponent'])
    hse=hub**(f[iden].attrs['h-scale-exponent'])
    cgscf=f[iden].attrs['CGSConversionFactor']*ase*hse
    dset=f[iden][st:fh]*cgscf
    return dset

#Based heavily on work by David Barnes
def find_files(sn):

    path='/scratch/nas_virgo/Cosmo-OWLS/AGN_TUNED_nu0_L400N1024_Planck'

    if os.path.isfile(path+'/particledata_'+sn+'/eagle_subfind_particles_'+sn+'.0.hdf5') == True:
        pd=path+'/particledata_'+sn+'/eagle_subfind_particles_'+sn+'.0.hdf5'
    sd=[]
    for x in os.listdir(path+'/groups_'+sn+'/'):
        if x.startswith('eagle_subfind_tab_'):
            sd.append(path+'/groups_'+sn+'/'+x)
    odr=[]
    for x in sd: odr.append(int(x[94:-5]))
    odr=np.array(odr)
    so=np.argsort(odr)
    del odr
    sd=np.array(sd)
    sd=sd[so]
    del so
    sd=list(sd)
    return [sd,pd]



 # for index_shift, index_start in enumerate(range(0, data_size, CHUNK_SIZE)):
                #     index_end = index_shift*(CHUNK_SIZE+1)-1 if (index_shift+1)*CHUNK_SIZE-1 < data_size else data_size - 1
                #     part_gn = h5file[f'/PartType{part_type}/GroupNumber'][index_start:index_end]
                #     part_gn_index = np.where(part_gn == self.centralFOF_groupNumber+1)[0]
                #     del part_gn
                #     part_coords = h5file[f'/PartType{part_type}/Coordinates'][index_start:index_end][part_gn_index]
                #     del part_gn_index
                #     coords = np.concatenate((coords, part_coords), axis=0)
                #     yield ((counter+1) / (data_size/CHUNK_SIZE))  # Give control back to decorator
                #     counter += 1

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

 HDF5 "groups_032/eagle_subfind_tab_032.0.hdf5" {
FILE_CONTENTS {
 group      /
 group      /Constants
 group      /FOF
 dataset    /FOF/ContaminationCount
 dataset    /FOF/ContaminationMass
 dataset    /FOF/FirstSubhaloID
 dataset    /FOF/GroupCentreOfPotential
 dataset    /FOF/GroupLength
 dataset    /FOF/GroupMass
 dataset    /FOF/GroupOffset
 dataset    /FOF/Group_M_Crit200
 dataset    /FOF/Group_M_Crit2500
 dataset    /FOF/Group_M_Crit500
 dataset    /FOF/Group_M_Mean200
 dataset    /FOF/Group_M_Mean2500
 dataset    /FOF/Group_M_Mean500
 dataset    /FOF/Group_M_TopHat200
 dataset    /FOF/Group_R_Crit200
 dataset    /FOF/Group_R_Crit2500
 dataset    /FOF/Group_R_Crit500
 dataset    /FOF/Group_R_Mean200
 dataset    /FOF/Group_R_Mean2500
 dataset    /FOF/Group_R_Mean500
 dataset    /FOF/Group_R_TopHat200
 dataset    /FOF/NumOfSubhalos
 group      /Header
 group      /IDs
 dataset    /IDs/ParticleID
 dataset    /IDs/Particle_Binding_Energy
 group      /Parameters
 group      /Parameters/ChemicalElements
 group      /Subhalo
 dataset    /Subhalo/CentreOfMass
 dataset    /Subhalo/CentreOfPotential
 dataset    /Subhalo/GasSpin
 dataset    /Subhalo/GroupNumber
 dataset    /Subhalo/HalfMassProjRad
 dataset    /Subhalo/HalfMassRad
 dataset    /Subhalo/IDMostBound
 dataset    /Subhalo/InertiaTensor
 dataset    /Subhalo/InitialMassWeightedBirthZ
 dataset    /Subhalo/InitialMassWeightedStellarAge
 dataset    /Subhalo/KineticEnergy
 dataset    /Subhalo/Mass
 dataset    /Subhalo/MassType
 dataset    /Subhalo/Mass_001kpc
 dataset    /Subhalo/Mass_003kpc
 dataset    /Subhalo/Mass_005kpc
 dataset    /Subhalo/Mass_010kpc
 dataset    /Subhalo/Mass_020kpc
 dataset    /Subhalo/Mass_030kpc
 dataset    /Subhalo/Mass_040kpc
 dataset    /Subhalo/Mass_050kpc
 dataset    /Subhalo/Mass_070kpc
 dataset    /Subhalo/Mass_100kpc
 group      /Subhalo/NSF
 group      /Subhalo/NSF/ElementAbundance
 dataset    /Subhalo/NSF/ElementAbundance/Carbon
 dataset    /Subhalo/NSF/ElementAbundance/Helium
 dataset    /Subhalo/NSF/ElementAbundance/Hydrogen
 dataset    /Subhalo/NSF/ElementAbundance/Iron
 dataset    /Subhalo/NSF/ElementAbundance/Magnesium
 dataset    /Subhalo/NSF/ElementAbundance/Neon
 dataset    /Subhalo/NSF/ElementAbundance/Nitrogen
 dataset    /Subhalo/NSF/ElementAbundance/Oxygen
 dataset    /Subhalo/NSF/ElementAbundance/Silicon
 dataset    /Subhalo/NSF/KineticEnergy
 dataset    /Subhalo/NSF/Mass
 dataset    /Subhalo/NSF/MassWeightedEntropy
 dataset    /Subhalo/NSF/MassWeightedTemperature
 dataset    /Subhalo/NSF/MetalMass
 dataset    /Subhalo/NSF/MetalMassSmoothed
 group      /Subhalo/NSF/SmoothedElementAbundance
 dataset    /Subhalo/NSF/Spin
 dataset    /Subhalo/NSF/ThermalEnergy
 dataset    /Subhalo/NSF/TotalEnergy
 dataset    /Subhalo/Parent
 group      /Subhalo/SF
 group      /Subhalo/SF/ElementAbundance
 dataset    /Subhalo/SF/ElementAbundance/Carbon
 dataset    /Subhalo/SF/ElementAbundance/Helium
 dataset    /Subhalo/SF/ElementAbundance/Hydrogen
 dataset    /Subhalo/SF/ElementAbundance/Iron
 dataset    /Subhalo/SF/ElementAbundance/Magnesium
 dataset    /Subhalo/SF/ElementAbundance/Neon
 dataset    /Subhalo/SF/ElementAbundance/Nitrogen
 dataset    /Subhalo/SF/ElementAbundance/Oxygen
 dataset    /Subhalo/SF/ElementAbundance/Silicon
 dataset    /Subhalo/SF/KineticEnergy
 dataset    /Subhalo/SF/Mass
 dataset    /Subhalo/SF/MassWeightedEntropy
 dataset    /Subhalo/SF/MassWeightedTemperature
 dataset    /Subhalo/SF/MetalMass
 dataset    /Subhalo/SF/MetalMassSmoothed
 group      /Subhalo/SF/SmoothedElementAbundance
 dataset    /Subhalo/SF/Spin
 dataset    /Subhalo/SF/ThermalEnergy
 dataset    /Subhalo/SF/TotalEnergy
 dataset    /Subhalo/StarFormationRate
 group      /Subhalo/Stars
 group      /Subhalo/Stars/ElementAbundance
 dataset    /Subhalo/Stars/ElementAbundance/Carbon
 dataset    /Subhalo/Stars/ElementAbundance/Helium
 dataset    /Subhalo/Stars/ElementAbundance/Hydrogen
 dataset    /Subhalo/Stars/ElementAbundance/Iron
 dataset    /Subhalo/Stars/ElementAbundance/Magnesium
 dataset    /Subhalo/Stars/ElementAbundance/Neon
 dataset    /Subhalo/Stars/ElementAbundance/Nitrogen
 dataset    /Subhalo/Stars/ElementAbundance/Oxygen
 dataset    /Subhalo/Stars/ElementAbundance/Silicon
 dataset    /Subhalo/Stars/KineticEnergy
 dataset    /Subhalo/Stars/Mass
 dataset    /Subhalo/Stars/MetalMass
 dataset    /Subhalo/Stars/MetalMassSmoothed
 group      /Subhalo/Stars/SmoothedElementAbundance
 dataset    /Subhalo/Stars/Spin
 dataset    /Subhalo/Stars/TotalEnergy
 dataset    /Subhalo/StellarInitialMass
 dataset    /Subhalo/StellarVelDisp
 dataset    /Subhalo/StellarVelDisp_HalfMassProjRad
 dataset    /Subhalo/SubLength
 dataset    /Subhalo/SubOffset
 dataset    /Subhalo/ThermalEnergy
 dataset    /Subhalo/TotalEnergy
 dataset    /Subhalo/Velocity
 dataset    /Subhalo/Vmax
 dataset    /Subhalo/VmaxRadius
 group      /Units
 }

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


 Header attrs
 "BoxSize"
 "ExpansionFactor"
 "Flag_Cooling"
 "Flag_DoublePrecision"
 "Flag_Feedback"
 "Flag_IC_Info"
 "Flag_Metals"
 "Flag_Sfr"
 "Flag_StellarAge"
 "HubbleParam"
 "MassTable"
 "NTask"
 "Ngroups"
 "Nids"
 "Nsubgroups"
 "NumFilesPerSnapshot"
 "NumPart_ThisFile"
 "NumPart_Total"
 "NumPart_Total_HighWord"
 "Omega0"
 "OmegaBaryon"
 "OmegaLambda"
 "Redshift"
 "RunLabel"
 "SendOffSetTask"
 "Time"
 "TotNgroups"
 "TotNids"
 "TotNsubgroups"
 """
