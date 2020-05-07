"""
------------------------------------------------------------------
FILE:   _cluster_retriever.py
AUTHOR: Edo Altamura
DATE:   12-11-2019
------------------------------------------------------------------
This file is an extension of the cluster.Cluster class. It provides
class methods for reading C-EAGLE data from the /cosma5 data system.
This file contains a mixin class, affiliated to cluster.Cluster.
Mixins are classes that have no data of their own — only methods —
so although you inherit them, you never have to call super() on them.
They working principle is based on OOP class inheritance.
-------------------------------------------------------------------
"""

from functools import wraps
import os
import h5py as h5
import numpy as np
from .memory import free_memory
from .progressbar import ProgressBar


def redshift_str2num(z: str):
    """
    Converts the redshift of the snapshot from text to numerical,
    in a format compatible with the file names.
    E.g. float z = 2.16 <--- str z = 'z002p160'.
    """
    z = z.strip('z').replace('p', '.')
    return round(float(z), 3)


def redshift_num2str(z: float):
    """
    Converts the redshift of the snapshot from numerical to
    text, in a format compatible with the file names.
    E.g. float z = 2.16 ---> str z = 'z002p160'.
    """
    z = round(z, 3)
    integer_z, decimal_z = str(z).split('.')
    integer_z = int(integer_z)
    decimal_z = int(decimal_z)
    return f"z{integer_z:0>3d}p{decimal_z:0<3d}"



class Mixin:

    #####################################################
    #													#
    #				D E C O R A T O R S  				#
    # 									 				#
    #####################################################

    def data_subject(**decorator_kwargs):
        """
        This decorator adds functionality to the data import functions.
        It dynamically creates attributes and filepaths, to be then
        passed onto the actual import methods.
        :param decorator_kwargs: subject = (str)
        :return: decorated function with predefined **kwargs
        The **Kwargs are dynamically allocated to the external methods.
        """

        # print("Reading ", decorator_kwargs['subject'], " files.")

        def wrapper(f):  # a wrapper for the function
            @wraps(f)
            def decorated_function(self, *args, **kwargs):  # the decorated function

                redshift_i = self.redshiftAllowed.index(self.redshift)
                redshift_index = self.zcat['z_IDNumber'][redshift_i]

                sbj_string = decorator_kwargs['subject'] + '_' + redshift_index

                if self.simulation_name == 'celr_e' or self.simulation_name == 'ceagle':
                    sbj_string = sbj_string + '_' + self.redshift

                elif (self.simulation_name == 'celr_b' or
                      self.simulation_name == 'macsis' or
                      self.simulation_name == 'bahamas'):
                    sbj_string = sbj_string


                file_dir = os.path.join(self.path_from_cluster_name(), sbj_string)
                try:
                    file_list = os.listdir(file_dir)
                except:
                    file_list = []

                if decorator_kwargs['subject'] == 'particledata':
                    prefix = 'eagle_subfind_particles_'
                elif decorator_kwargs['subject'] == 'groups':
                    prefix = 'eagle_subfind_tab_'
                elif decorator_kwargs['subject'] == 'snapshot':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'snipshot':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'hsmldir':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
                elif decorator_kwargs['subject'] == 'groups_snip':
                    raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")

                # Transfer function state into the **kwargs
                # These **kwargs are accessible to the decorated class methods
                kwargs['subject'] = decorator_kwargs['subject']
                kwargs['file_dir'] = file_dir
                kwargs['file_list'] = [x for x in file_list if x.startswith(prefix)]
                kwargs['file_list_sorted'] = sorted([os.path.join(file_dir, file) for file in kwargs['file_list']])
                # print(kwargs['file_list_sorted'])

                return f(self, *args, **kwargs)

            return decorated_function

        return wrapper

    #####################################################
    #													#
    #					D A T A   						#
    # 				M A N A G E M E N T 				#
    #													#
    #####################################################
    @data_subject(subject="groups")
    def groups_fileDir(self, **kwargs):
        return kwargs['file_dir']

    @data_subject(subject="particledata")
    def partdata_fileDir(self, **kwargs):
        return kwargs['file_dir']

    @data_subject(subject="groups")
    def groups_filePaths(self, **kwargs):
        return kwargs['file_list_sorted']

    @data_subject(subject="particledata")
    def partdata_filePaths(self, **kwargs):
        return kwargs['file_list_sorted']

    @data_subject(subject="groups")
    def group_centre_of_potential(self, *args, **kwargs):
        """
        AIM: reads the FoF group central of potential from the path and file given
        RETURNS: type = np.array of 3 doubles
        ACCESS DATA: e.g. group_CoP[0] for getting the x value
        """
        dat_index = self.clusterID if self.simulation_name is 'bahamas' else 0
        with h5.File(kwargs['file_list_sorted'][0], 'r') as h5file:
            pos = h5file['/FOF/GroupCentreOfPotential'][dat_index]
            if not self.comovingframe:
                pos = self.comoving_length(pos)
            free_memory(['pos'], invert=True)
        return pos

    @data_subject(subject="groups")
    def group_r200(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        dat_index = self.clusterID if self.simulation_name is 'bahamas' else 0
        with h5.File(kwargs['file_list_sorted'][0], 'r') as h5file:
            h5dset = h5file["/FOF/Group_R_Crit200"]
            r200c = h5dset[...][0]
            if not self.comovingframe:
                r200c = self.comoving_length(r200c)
            free_memory(['r200c'], invert=True)
        return r200c

    @data_subject(subject="groups")
    def group_r500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][0], 'r') as h5file:
            h5dset = h5file["/FOF/Group_R_Crit500"]
            r500c = h5dset[...][0]

            if not self.comovingframe:
                r500c = self.comoving_length(r500c)
            free_memory(['r500c'], invert=True)
        return r500c

    @data_subject(subject="groups")
    def group_r2500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/FOF/Group_R_Crit2500"]
        temp = h5dset[...]
        h5file.close()
        r2500c = temp[0]
        if not self.comovingframe:
            r2500c = self.comoving_length(r2500c)
        free_memory(['r2500c'], invert=True)
        return r2500c

    @data_subject(subject="groups")
    def group_mass(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/FOF/GroupMass"]
        temp = h5dset[...]
        h5file.close()
        m_tot = temp[0]
        if not self.comovingframe:
            m_tot = self.comoving_mass(m_tot)
        free_memory(['m_tot'], invert=True)
        return m_tot

    @data_subject(subject="groups")
    def group_M200(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/FOF/Group_M_Crit200"]
        temp = h5dset[...]
        h5file.close()
        m200 = temp[0]
        if not self.comovingframe:
            m200 = self.comoving_mass(m200)
        free_memory(['m200'], invert=True)
        return m200

    @data_subject(subject="groups")
    def group_M500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/FOF/Group_M_Crit500"]
        temp = h5dset[...]
        h5file.close()
        m500 = temp[0]
        if not self.comovingframe:
            m500 = self.comoving_mass(m500)
        free_memory(['m500'], invert=True)
        return m500

    @data_subject(subject="groups")
    def group_M2500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/FOF/Group_M_Crit2500"]
        temp = h5dset[...]
        h5file.close()
        m2500 = temp[0]
        if not self.comovingframe:
            m2500 = self.comoving_mass(m2500)
        free_memory(['m2500'], invert=True)
        return m2500

    @data_subject(subject="groups")
    def NumOfSubhalos(self, *args, central_FOF=None, **kwargs):
        """
        AIM: retrieves the redshift of the file
        RETURNS: type = int

        NOTES: there is no file for the FOF group number array. Instead,
                there is an array in /FOF for the number of subhalos in each
                FOF group. Used to gather each subgroup number
        """
        if central_FOF:
            h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
            h5dset = h5file["/FOF/NumOfSubhalos"]
            attr_value = h5dset[...]
            h5file.close()
            Ngroups = attr_value[0]
            free_memory(['Ngroups'], invert=True)
            return Ngroups

        elif central_FOF is None or not central_FOF:
            Ngroups = np.zeros(0, dtype=np.int)
            for path in kwargs['file_list_sorted']:
                h5file = h5.File(path, 'r')
                h5dset = h5file["/FOF/NumOfSubhalos"]
                attr_value = h5dset[...]
                h5file.close()
                Ngroups = np.concatenate((Ngroups, attr_value), axis=0)
            free_memory(['Ngroups'], invert=True)
            return Ngroups

    @data_subject(subject="groups")
    def SubGroupNumber(self, *args, central_FOF=None, **kwargs):
        """
        AIM: reads the group number of subgroups from the path and file given
        RETURNS: type = 1/2D np.array

        if central_FOF:
            returns []

        """
        if central_FOF:
            # Build the sgn list artificially: less overhead in opening files
            n_subhalos = self.NumOfSubhalos(*args, central_FOF=True, **kwargs)
            sgn_list = np.linspace(0, n_subhalos - 1, n_subhalos, dtype=np.int)
            free_memory(['sgn_list'], invert=True)
            return sgn_list

        elif central_FOF is None or not central_FOF:
            sgn_list = np.zeros(0, dtype=np.int)
            for path in kwargs['file_list_sorted']:
                h5file = h5.File(path, 'r')
                h5dset = h5file["/Subhalo/SubGroupNumber"]
                sgn_sublist = h5dset[...]
                h5file.close()
                sgn_list = np.concatenate((sgn_list, sgn_sublist), axis=0)
            # Check that the len of array is == total no of subhalos
            assert np.sum(self.NumOfSubhalos(*args, central_FOF=False, **kwargs)) == sgn_list.__len__()
            free_memory(['sgn_list'], invert=True)
            return sgn_list

    @data_subject(subject="groups")
    def subgroups_centre_of_potential(self, *args, **kwargs):
        """
        AIM: reads the subgroups central of potential from the path and file given
        RETURNS: type = 2D np.array
        FORMAT:	 sub#	  x.sub_CoP 	y.sub_CoP     z.sub_CoP
                    0  [[			,				,			],
                    1	[			,				,			],
                    2	[			,				,			],
                    3	[			,				,			],
                    4	[			,				,			],
                    5	[			,				,			],
                    .						.
                    .						.
                    .						.					]]

        """
        pos = np.zeros((0, 3), dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/CentreOfPotential']
            sub_CoP = hd5set[...]
            h5file.close()
            pos = np.concatenate((pos, sub_CoP), axis=0)
            free_memory(['pos'], invert=True)

        if not self.comovingframe:
            pos = self.comoving_length(pos)
        return pos

    @data_subject(subject="groups")
    def subgroups_centre_of_mass(self, *args, **kwargs):
        """
        AIM: reads the subgroups central of mass from the path and file given
        RETURNS: type = 2D np.array
        FORMAT:	 sub#	  x.sub_CoM 	y.sub_CoM     z.sub_CoM
                    0  [[			,				,			],
                    1	[			,				,			],
                    2	[			,				,			],
                    3	[			,				,			],
                    4	[			,				,			],
                    5	[			,				,			],
                    .						.
                    .						.
                    .						.					]]

        """

        pos = np.zeros((0, 3), dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/CentreOfMass']
            sub_CoM = hd5set[...]
            h5file.close()
            pos = np.concatenate((pos, sub_CoM), axis=0)
            free_memory(['pos'], invert=True)

        if not self.comovingframe:
            pos = self.comoving_length(pos)
        return pos

    @data_subject(subject="groups")
    def subgroups_velocity(self, *args, **kwargs):
        """
        AIM: reads the subgroups 3d velocities from the path and file given
        RETURNS: type = 2D np.array
        FORMAT:	 sub#	    vx.sub 	 	 vy.sub       	vz.sub
                    0  [[			,				,			],
                    1	[			,				,			],
                    2	[			,				,			],
                    3	[			,				,			],
                    4	[			,				,			],
                    5	[			,				,			],
                    .						.
                    .						.
                    .						.					]]

        """

        vel = np.zeros((0, 3), dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/Velocity']
            sub_v = hd5set[...]
            h5file.close()
            vel = np.concatenate((vel, sub_v), axis=0)
            free_memory(['vel'], invert=True)

        if not self.comovingframe:
            vel = self.comoving_velocity(vel)
        return vel

    @data_subject(subject="groups")
    def subgroups_mass(self, *args, **kwargs):
        """
        AIM: reads the subgroups masses from the path and file given
        RETURNS: type = 1D np.array
        """

        mass = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/Mass']
            sub_m = hd5set[...]
            h5file.close()
            mass = np.concatenate((mass, sub_m))
            free_memory(['mass'], invert=True)

        if not self.comovingframe:
            mass = self.comoving_mass(mass)
        return mass

    @data_subject(subject="groups")
    def subgroups_kin_energy(self, *args, **kwargs):
        """
        AIM: reads the subgroups kinetic energy from the path and file given
        RETURNS: type = 1D np.array
        """
        kin_energy = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/KineticEnergy']
            sub_ke = hd5set[...]
            h5file.close()
            kin_energy = np.concatenate((kin_energy, sub_ke), axis=0)
            free_memory(['kin_energy'], invert=True)

        if not self.comovingframe:
            kin_energy = self.comoving_kinetic_energy(kin_energy)
        return kin_energy

    @data_subject(subject="groups")
    def subgroups_therm_energy(self, *args, **kwargs):
        """
        AIM: reads the subgroups thermal energy from the path and file given
        RETURNS: type = 1D np.array
        """
        therm_energy = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/Subhalo/ThermalEnergy']
            sub_th = hd5set[...]
            h5file.close()
            therm_energy = np.concatenate((therm_energy, sub_th), axis=0)
            free_memory(['therm_energy'], invert=True)
        return therm_energy

    @data_subject(subject="particledata")
    def group_number_part(self, part_type, *args, **kwargs):
        """
        RETURNS: np.array
        """
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        group_number = np.zeros(0, dtype=np.int)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType' + part_type + '/GroupNumber']
            sub_gn = hd5set[...]
            h5file.close()
            group_number = np.concatenate((group_number, sub_gn), axis=0)
            assert group_number.__len__() > 0, "Array is empty."
        return group_number

    @ProgressBar()
    @data_subject(subject="particledata")
    def subgroup_number_part(self, part_type, *args, **kwargs):

        if len(part_type) > 1:
            part_type = self.particle_type_conversion[part_type]

        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        sub_group_number = np.zeros(0, dtype=np.int)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType' + part_type + '/SubGroupNumber']
            sub_gn = hd5set[...]
            h5file.close()
            sub_group_number = np.concatenate((sub_group_number, sub_gn), axis=0)
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        return sub_group_number

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_coordinates(self, part_type, *args, **kwargs):
        """
        RETURNS: 2D np.array
        """
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        pos = np.zeros((0, 3), dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType' + part_type + '/Coordinates']
            sub_pos = hd5set[...]
            h5file.close()
            pos = np.concatenate((pos, sub_pos), axis=0)
            free_memory(['pos'], invert=True)
            assert pos.__len__() > 0, "Array is empty."
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        if not self.comovingframe:
            pos = self.comoving_length(pos)
        return pos

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_velocity(self, part_type, *args, **kwargs):
        """
        RETURNS: 2D np.array
        """
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        part_vel = np.zeros((0, 3), dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType' + part_type + '/Velocity']
            sub_vel = hd5set[...]
            h5file.close()
            part_vel = np.concatenate((part_vel, sub_vel), axis=0)
            free_memory(['part_vel'], invert=True)
            assert part_vel.__len__() > 0, "Array is empty."
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        if not self.comovingframe:
            part_vel = self.comoving_velocity(part_vel)
        return part_vel

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_masses(self, part_type, *args, **kwargs):
        """
        RETURNS: 2D np.array
        """
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        if part_type == '1':
            part_mass = np.ones(self.DM_NumPart_Total()) * self.DM_particleMass()
        else:
            counter = 0
            length_operation = len(kwargs['file_list_sorted'])
            part_mass = np.zeros(0, dtype=np.float)
            for path in kwargs['file_list_sorted']:
                h5file = h5.File(path, 'r')
                hd5set = h5file['/PartType' + part_type + '/Mass']
                sub_m = hd5set[...]
                h5file.close()
                part_mass = np.concatenate((part_mass, sub_m), axis=0)
                free_memory(['part_mass'], invert=True)
                yield ((counter + 1) / length_operation)  # Give control back to decorator
                counter += 1

        assert part_mass.__len__() > 0, "Array is empty."

        if not self.comovingframe:
            part_mass = self.comoving_mass(part_mass)

        return part_mass

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_temperature(self, *args, **kwargs):
        """
        RETURNS: 1D np.array
        """
        # Check that we are extracting the temperature of gas particles
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        temperature = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType0/Temperature']
            sub_T = hd5set[...]
            h5file.close()
            temperature = np.concatenate((temperature, sub_T), axis=0)
            free_memory(['temperature'], invert=True)
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        assert temperature.__len__() > 0, "Array is empty."
        return temperature

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_SPH_density(self, *args, **kwargs):
        """
        RETURNS: 1D np.array
        """
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        densitySPH = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType0/Density']
            sub_den = hd5set[...]
            h5file.close()
            densitySPH = np.concatenate((densitySPH, sub_den), axis=0)
            free_memory(['densitySPH'], invert=True)
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        assert densitySPH.__len__() > 0, "Array is empty."

        if not self.comovingframe:
            densitySPH = self.comoving_density(densitySPH)

        return densitySPH

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_SPH_smoothinglength(self, *args, **kwargs):
        """
        RETURNS: 1D np.array
        """
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        smoothinglength = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType0/SmoothingLength']
            sub_den = hd5set[...]
            h5file.close()
            smoothinglength = np.concatenate((smoothinglength, sub_den), axis=0)
            free_memory(['smoothinglength'], invert=True)
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        assert smoothinglength.__len__() > 0, "Array is empty."

        if not self.comovingframe:
            smoothinglength = self.comoving_length(smoothinglength)

        return smoothinglength

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_metallicity(self, *args, **kwargs):
        """
        RETURNS: 1D np.array
        """
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        metallicity = np.zeros(0, dtype=np.float)
        for path in kwargs['file_list_sorted']:
            h5file = h5.File(path, 'r')
            hd5set = h5file['/PartType0/Metallicity']
            sub_Z = hd5set[...]
            h5file.close()
            metallicity = np.concatenate((metallicity, sub_Z), axis=0)
            free_memory(['metallicity'], invert=True)
            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

        if not self.comovingframe:
            raise("Metallicity not yet implemented for comoving -> physical frame conversion.")

        return metallicity

    @data_subject(subject="particledata")
    def extract_header_attribute(self, element_number, *args, **kwargs):
        # Import data from hdf5 file
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/Header"]
        attr_name = list(h5dset.attrs.keys())[element_number]
        attr_value = list(h5dset.attrs.values())[element_number]
        h5file.close()
        return attr_name, attr_value

    @data_subject(subject="particledata")
    def extract_header_attribute_name(self, element_name, *args, **kwargs):
        # Import data from hdf5 file
        h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
        h5dset = h5file["/Header"]
        attr_name = h5dset.attrs.get(element_name, default=None)
        attr_value = h5dset.attrs.get(element_name, default=None)
        h5file.close()
        return attr_name, attr_value