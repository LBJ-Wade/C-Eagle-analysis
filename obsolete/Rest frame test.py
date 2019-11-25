import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile


# Specify object
simulation_type = 'gas'
num_halo = 0
redshift = 0.5

# Import data
path = 		extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
file = 		extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
r200 = 		extract.group_r200(path, file)
group_CoP = extract.group_centre_of_potential(path, file)
file = 		extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
part_type = extract.particle_type('gas')
#mass = 		extract.particle_temperature(path, file, part_type)
coordinates = extract.particle_coordinates(path, file, part_type)
#temperature = extract.particle_temperature(path, file, part_type)

velocity = extract.particle_velocity(path, file, part_type)
rest_zero = profile.cluster_average_velocity(velocity)
rest_one, _ = profile.cluster_average_momentum(path, file, part_type)
rest_two, _ = profile.total_mass_rest_frame(path, file)


print(path)
print("\nPure average of velocities") 
print(rest_zero)
print("\nAverage of momenta (mass weighted) for 1 part_type") 
print(rest_one)
print("\nAverage of momenta (mass weighted) for all part_types") 
print(rest_two)


# Filter 
print(' - - - - - - - - - \nEnd of file.')

'''
80.757256 144.66231 202.04544
'''