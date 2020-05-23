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
import re
import h5py as h5
import numpy as np
from .memory import free_memory
from .progressbar import ProgressBar

CHUNK_SIZE = 1000000

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


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


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
                file_list = [x for x in file_list if x.startswith(prefix)]
                file_list.sort(key=natural_sort_key)
                kwargs['subject'] = decorator_kwargs['subject']
                kwargs['file_dir'] = file_dir
                kwargs['file_list'] = file_list
                kwargs['file_list_sorted'] = [os.path.join(file_dir, file) for file in file_list]
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
    def file_group_indexify(self, **kwargs):
        if self.simulation_name == 'bahamas':
            Ngroups = 0
            file_counter = 0
            element_counter = self.halo_num_catalogue_contiguous[self.clusterID]
            while True:
                with h5.File(kwargs['file_list_sorted'][file_counter], 'r') as h5file:
                    Ngroups += h5file['Header'].attrs['Ngroups']
                    if element_counter < h5file['Header'].attrs['Ngroups']:
                        break
                    else:
                        file_counter += 1
                        element_counter -= h5file['Header'].attrs['Ngroups']
            return file_counter, element_counter
        else:
            return 0, 0

    @data_subject(subject="groups")
    def group_centre_of_potential(self, *args, **kwargs):
        """
        AIM: reads the FoF group central of potential from the path and file given
        RETURNS: type = np.array of 3 doubles
        ACCESS DATA: e.g. group_CoP[0] for getting the x value
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            pos = h5file['/FOF/GroupCentreOfPotential'][self.groupfof_counter]
        pos = pos if self.comovingframe else self.comoving_length(pos)
        free_memory(['pos'], invert=True)
        return pos

    @data_subject(subject="groups")
    def group_r200(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            r200c = h5file['/FOF/Group_R_Crit200'][self.groupfof_counter]
        r200c = r200c if self.comovingframe else self.comoving_length(r200c)
        free_memory(['r200c'], invert=True)
        return r200c

    @data_subject(subject="groups")
    def group_r500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            r500c = h5file['/FOF/Group_R_Crit500'][self.groupfof_counter]
        r500c = r500c if self.comovingframe else self.comoving_length(r500c)
        free_memory(['r500c'], invert=True)
        return r500c

    @data_subject(subject="groups")
    def group_r2500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            r2500c = h5file['/FOF/Group_R_Crit2500'][self.groupfof_counter]
        r2500c = r2500c if self.comovingframe else self.comoving_length(r2500c)
        free_memory(['r2500c'], invert=True)
        return r2500c

    @data_subject(subject="groups")
    def group_mass(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            m_fof = h5file['/FOF/GroupMass'][self.groupfof_counter]
        m_fof = m_fof if self.comovingframe else self.comoving_mass(m_fof)
        free_memory(['m_fof'], invert=True)
        return m_fof

    @data_subject(subject="groups")
    def group_M200(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            m200c = h5file['/FOF/Group_M_Crit200'][self.groupfof_counter]
        m200c = m200c if self.comovingframe else self.comoving_mass(m200c)
        free_memory(['m200c'], invert=True)
        return m200c

    @data_subject(subject="groups")
    def group_M500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            m500c = h5file['/FOF/Group_M_Crit500'][self.groupfof_counter]
        m500c = m500c if self.comovingframe else self.comoving_mass(m500c)
        free_memory(['m500c'], invert=True)
        return m500c

    @data_subject(subject="groups")
    def group_M2500(self, *args, **kwargs):
        """
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            m2500c = h5file['/FOF/Group_M_Crit2500'][self.groupfof_counter]
        m2500c = m2500c if self.comovingframe else self.comoving_mass(m2500c)
        free_memory(['m2500c'], invert=True)
        return m2500c

    @data_subject(subject="groups")
    def NumOfSubhalos(self, *args, **kwargs):
        """
        AIM: retrieves the redshift of the file
        RETURNS: type = int

        NOTES: there is no file for the FOF group number array. Instead,
                there is an array in /FOF for the number of subhalos in each
                FOF group. Used to gather each subgroup number
                Nsubgroups
        """
        with h5.File(kwargs['file_list_sorted'][self.file_counter], 'r') as h5file:
            Nsubgroups = h5file['/FOF/NumOfSubhalos'][self.groupfof_counter]
        free_memory(['Nsubgroups'], invert=True)
        return Nsubgroups

    @data_subject(subject="groups")
    def subhalo_groupNumber(self, *args, **kwargs):
        """
        SUBGROUP NUMBER INDEX

        AIM: reads the group number of subgroups from the path and file given
        RETURNS: type = 1/2D np.array
        """
        subhalo_groupNumber = np.zeros(0, dtype=np.int)
        base_index_shift = 0
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                sub_gn = np.where(h5file['Subhalo/GroupNumber'][:] == self.clusterID)[0] + base_index_shift
                subhalo_groupNumber = np.concatenate((subhalo_groupNumber, sub_gn))
                base_index_shift += h5file['Subhalo/GroupNumber'].size

        free_memory(['subhalo_groupNumber'], invert=True)
        return subhalo_groupNumber

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
        CoP = np.zeros((0, 3), dtype=np.float)
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_CoP = h5file['Subhalo/CentreOfPotential'][subhalo_gn_index]
                CoP = np.concatenate((CoP, sub_CoP))
                
        CoP = CoP if self.comovingframe else self.comoving_length(CoP)
        free_memory(['CoP'], invert=True)
        return CoP

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
        CoM = np.zeros((0, 3), dtype=np.float)
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_CoM = h5file['Subhalo/CentreOfMass'][subhalo_gn_index]
                CoM = np.concatenate((CoM, sub_CoM))

        CoM = CoM if self.comovingframe else self.comoving_length(CoM)
        free_memory(['CoM'], invert=True)
        return CoM

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
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_vel = h5file['Subhalo/Velocity'][subhalo_gn_index]
                vel = np.concatenate((vel, sub_vel))

        vel = vel if self.comovingframe else self.comoving_velocity(vel)
        free_memory(['vel'], invert=True)
        return vel

    @data_subject(subject="groups")
    def subgroups_mass(self, *args, **kwargs):
        """
        AIM: reads the subgroups masses from the path and file given
        RETURNS: type = 1D np.array
        """
        mass = np.zeros(0, dtype=np.float)
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_mass = h5file['Subhalo/Mass'][subhalo_gn_index]
                mass = np.concatenate((mass, sub_mass))

        mass = mass if self.comovingframe else self.comoving_mass(mass)
        free_memory(['mass'], invert=True)
        return mass

    @data_subject(subject="groups")
    def subgroups_kin_energy(self, *args, **kwargs):
        """
        AIM: reads the subgroups kinetic energy from the path and file given
        RETURNS: type = 1D np.array
        """
        kinetic = np.zeros(0, dtype=np.float)
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_kinetic = h5file['Subhalo/KineticEnergy'][subhalo_gn_index]
                kinetic = np.concatenate((kinetic, sub_kinetic))

        kinetic = kinetic if self.comovingframe else self.comoving_kinetic_energy(kinetic)
        free_memory(['kinetic'], invert=True)
        return kinetic

    @data_subject(subject="groups")
    def subgroups_therm_energy(self, *args, **kwargs):
        """
        AIM: reads the subgroups thermal energy from the path and file given
        RETURNS: type = 1D np.array
        """
        thermal = np.zeros(0, dtype=np.float)
        for file in kwargs['file_list_sorted']:
            with h5.File(file, 'r') as h5file:
                subhalo_gn_index = np.where(h5file['Subhalo/GroupNumber'][:] == self.centralFOF_groupNumber)[0]
                sub_thermal = h5file['Subhalo/ThermalEnergy'][subhalo_gn_index]
                thermal = np.concatenate((thermal, sub_thermal))

        free_memory(['thermal'], invert=True)
        return thermal

    @ProgressBar()
    @data_subject(subject="particledata")
    def group_number_part(self, part_type, *args, **kwargs):
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        group_number = np.zeros(0, dtype=np.int)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    group_number = np.concatenate((group_number, part_gn_index), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1
                    # data_size = h5file['Header'].attrs['NumPart_ThisFile'][int(part_type)]
                    # for index_shift, index_start in enumerate(range(0, data_size, CHUNK_SIZE)):
                    #     index_end = index_shift*(CHUNK_SIZE+1)-1 if (index_shift+1)*CHUNK_SIZE-1 < data_size else data_size-1
                    #     part_gn = h5file[f'/PartType{part_type}/GroupNumber'][index_start:index_end]
                    #     part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0] + index_start
                    #     group_number = np.concatenate((group_number, part_gn_index), axis=0)
                    #     yield ((counter + 1) / (length_operation*data_size/CHUNK_SIZE))  # Give control back to decorator
                    #     counter += 1

        else:
            base_index_shift = 0
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0] + base_index_shift
                    group_number = np.concatenate((group_number, part_gn_index), axis=0)
                    base_index_shift += h5file['Header'].attrs['NumPart_ThisFile'][int(part_type)]
                    yield ((counter + 1) / length_operation)  # Give control back to decorator
                    counter += 1

        free_memory(['group_number'], invert=True)
        assert group_number.__len__() > 0, "Array is empty."
        return group_number

    @ProgressBar()
    @data_subject(subject="particledata")
    def subgroup_number_part(self, part_type, *args, **kwargs):
        if len(part_type) > 1:
            part_type = self.particle_type_conversion[part_type]

        assert hasattr(self, f'partType{part_type}_groupnumber')
        part_gn_index = getattr(self, f'partType{part_type}_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        subgroup_number = np.zeros(0, dtype=np.int)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_sgn = h5file[f'/PartType{part_type}/SubGroupNumber'][part_gn_index]
                    subgroup_number = np.concatenate((subgroup_number, part_sgn), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_sgn = h5file[f'/PartType{part_type}/SubGroupNumber'][:][part_gn_index]
                    subgroup_number = np.concatenate((subgroup_number, part_sgn), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        free_memory(['subgroup_number'], invert=True)
        assert subgroup_number.__len__() > 0, "Array is empty."
        return subgroup_number

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_coordinates(self, part_type, *args, **kwargs):
        if len(part_type) > 1:
            part_type = self.particle_type_conversion[part_type]

        assert hasattr(self, f'partType{part_type}_groupnumber')
        part_gn_index = getattr(self, f'partType{part_type}_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        coords = np.zeros((0, 3), dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    boxsize = h5file['Header'].attrs['BoxSize']
                    part_coords = h5file[f'/PartType{part_type}/Coordinates'][part_gn_index]
                    coords = np.concatenate((coords, part_coords), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

                ## Periodic boundary wrapping
                for coord_axis in range(3):
                    # Right boundary
                    if self.centre_of_potential[coord_axis] + 5*self.r200 > boxsize:
                        beyond_index = np.where(coords[:, coord_axis] < boxsize/2)[0]
                        coords[beyond_index, coord_axis] = coords[beyond_index, coord_axis] + boxsize
                        del beyond_index

                    # Left boundary
                    elif self.centre_of_potential[coord_axis] - 5*self.r200 < 0.:
                        beyond_index = np.where(coords[:, coord_axis] > boxsize/2)[0]
                        coords[beyond_index, coord_axis] = coords[beyond_index, coord_axis] - boxsize
                        del beyond_index

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_coords = h5file[f'/PartType{part_type}/Coordinates'][:][part_gn_index]
                    coords = np.concatenate((coords, part_coords), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        free_memory(['coords'], invert=True)
        coords = coords if self.comovingframe else self.comoving_length(coords)
        assert len(coords) > 0, "Array is empty."
        return coords

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_velocity(self, part_type, *args, **kwargs):
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        assert hasattr(self, f'partType{part_type}_groupnumber')
        part_gn_index = getattr(self, f'partType{part_type}_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        vel = np.zeros((0, 3), dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_vel = h5file[f'/PartType{part_type}/Velocity'][part_gn_index]
                    vel = np.concatenate((vel, part_vel), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_vel = h5file[f'/PartType{part_type}/Velocity'][:][part_gn_index]
                    vel = np.concatenate((vel, part_vel), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        free_memory(['coords'], invert=True)
        vel = vel if self.comovingframe else self.comoving_velocity(vel)
        assert vel.__len__() > 0, "Array is empty."
        return vel

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_masses(self, part_type, *args, **kwargs):
        if part_type.__len__() > 1:
            part_type = self.particle_type_conversion[part_type]

        assert hasattr(self, f'partType{part_type}_groupnumber')
        part_gn_index = getattr(self, f'partType{part_type}_groupnumber')

        if part_type == '1':
            with h5.File(kwargs['file_list_sorted'][0], 'r') as h5file:
                particle_mass_DM = h5file['Header'].attrs['MassTable'][1]
            mass = np.ones(len(part_gn_index), dtype=np.float) * particle_mass_DM
        else:
            counter = 0
            length_operation = len(kwargs['file_list_sorted'])
            mass = np.zeros(0, dtype=np.float)

            if self.simulation_name is 'bahamas':
                for file in kwargs['file_list_sorted']:
                    with h5.File(file, 'r') as h5file:
                        part_mass = h5file[f'/PartType{part_type}/Mass'][part_gn_index]
                        mass = np.concatenate((mass, part_mass), axis=0)
                        yield ((counter + 1) / (length_operation))  # Give control back to decorator
                        counter += 1

            else:
                for file in kwargs['file_list_sorted']:
                    with h5.File(file, 'r') as h5file:
                        part_gn = h5file[f'/PartType{part_type}/GroupNumber'][:]
                        part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                        part_mass = h5file[f'/PartType{part_type}/Mass'][:][part_gn_index]
                        mass = np.concatenate((mass, part_mass), axis=0)
                        yield ((counter + 1) / (length_operation))  # Give control back to decorator
                        counter += 1

        free_memory(['mass'], invert=True)
        mass = mass if self.comovingframe else self.comoving_mass(mass)
        assert mass.__len__() > 0, "Array is empty."
        return mass

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_temperature(self, *args, **kwargs):
        """
        RETURNS: 1D np.array
        """
        # Check that we are extracting the temperature of gas particles
        assert hasattr(self, f'partType0_groupnumber')
        part_gn_index = getattr(self, f'partType0_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        temperature = np.zeros(0, dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_temperature = h5file[f'/PartType0/Temperature'][part_gn_index]
                    temperature = np.concatenate((temperature, part_temperature), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType0/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_temperature = h5file[f'/PartType0/Temperature'][:][part_gn_index]
                    temperature = np.concatenate((temperature, part_temperature), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        assert temperature.__len__() > 0, "Array is empty."
        return temperature

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_SPH_density(self, *args, **kwargs):

        assert hasattr(self, f'partType0_groupnumber')
        part_gn_index = getattr(self, f'partType0_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        density = np.zeros(0, dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_density = h5file[f'/PartType0/Density'][part_gn_index]
                    density = np.concatenate((density, part_density), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType0/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_density = h5file[f'/PartType0/Density'][:][part_gn_index]
                    density = np.concatenate((density, part_density), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        assert density.__len__() > 0, "Array is empty."
        density = density if self.comovingframe else self.comoving_density(density)
        return density

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_SPH_smoothinglength(self, *args, **kwargs):
        assert hasattr(self, f'partType0_groupnumber')
        part_gn_index = getattr(self, f'partType0_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        smoothinglength = np.zeros(0, dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_smoothinglength = h5file[f'/PartType0/SmoothingLength'][part_gn_index]
                    smoothinglength = np.concatenate((smoothinglength, part_smoothinglength), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType0/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_smoothinglength = h5file[f'/PartType0/SmoothingLength'][:][part_gn_index]
                    smoothinglength = np.concatenate((smoothinglength, part_smoothinglength), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        assert smoothinglength.__len__() > 0, "Array is empty."
        smoothinglength = smoothinglength if self.comovingframe else self.comoving_length(smoothinglength)
        return smoothinglength

    @ProgressBar()
    @data_subject(subject="particledata")
    def particle_metallicity(self, *args, **kwargs):
        assert hasattr(self, f'partType0_groupnumber')
        part_gn_index = getattr(self, f'partType0_groupnumber')
        counter = 0
        length_operation = len(kwargs['file_list_sorted'])
        metallicity = np.zeros(0, dtype=np.float)

        if self.simulation_name is 'bahamas':
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_metallicity = h5file[f'/PartType0/Metallicity'][part_gn_index]
                    metallicity = np.concatenate((metallicity, part_metallicity), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        else:
            for file in kwargs['file_list_sorted']:
                with h5.File(file, 'r') as h5file:
                    part_gn = h5file[f'/PartType0/GroupNumber'][:]
                    part_gn_index = np.where(part_gn == self.centralFOF_groupNumber)[0]
                    part_metallicity = h5file[f'/PartType0/Metallicity'][:][part_gn_index]
                    metallicity = np.concatenate((metallicity, part_metallicity), axis=0)
                    yield ((counter + 1) / (length_operation))  # Give control back to decorator
                    counter += 1

        assert metallicity.__len__() > 0, "Array is empty."
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