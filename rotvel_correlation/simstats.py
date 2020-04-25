"""
------------------------------------------------------------------
FILE:   rotvel_correlation/sinstats.py
AUTHOR: Edo Altamura
DATE:   21-04-2020
------------------------------------------------------------------
The save python package generates partial raw data from the simulation
data. The pull.py file accesses and reads these data and returns
then in a format that can be used for plotting, further processing
etc.
-------------------------------------------------------------------
"""
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import sys
import os
import warnings
import itertools
from typing import Union, Dict, List
import numpy as np
import pandas as pd
import h5py

exec(open(os.path.abspath(os.path.join(
		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from import_toolkit.progressbar import ProgressBar
from read import pull


class Simstats:
	"""
	Simstats class contains methods for generating files with statistics for all simulation clusters
	(all ID and all redshifts) with different parameters
	"""
	cols = [
			'cluster_id',
			'redshift_float',
			'R_2500_crit',
			'R_500_crit',
			'R_200_crit',
			'R_aperture',
			'M_2500_crit',
			'M_500_crit',
			'M_200_crit',
			'M_aperture_T',
			'M_aperture_0',
			'M_aperture_1',
			'M_aperture_4',
			'rotTvelT',
			'rot0rot4',
			'rot1rot4',
			'vel0vel1',
			'peculiar_velocity_T_magnitude',
			'peculiar_velocity_0_magnitude',
			'peculiar_velocity_1_magnitude',
			'peculiar_velocity_4_magnitude',
			'angular_momentum_T_magnitude',
			'angular_momentum_0_magnitude',
			'angular_momentum_1_magnitude',
			'angular_momentum_4_magnitude',
			'kinetic_energy_T',
			'kinetic_energy_0',
			'kinetic_energy_1',
			'thermal_energy',
			'substructure_mass_T',
			'dynamical_merging_index_T',
			'thermodynamic_merging_index_T',
			'substructure_fraction_T'
	]

	labels_tex = [
			r'Cluster ID',
			r'Redshift',
			r'$R_{2500,crit}$\quad[Mpc]',
			r'$R_{500,crit}$\quad[Mpc]',
			r'$R_{200,crit}$\quad[Mpc]',
			r'$R_\mathrm{aperture}$\quad[Mpc]',
			r'$M_{2500,crit}$\quad[$M_\odot$]',
			r'$M_{500,crit}$\quad[$M_\odot$]',
			r'$M_{200,crit}$\quad[$M_\odot$]',
			r'$M_{aperture}$\quad[$M_\odot$]',
			r'$M_\mathrm{aperture}^\mathrm{(gas)}$\quad[$M_\odot$]',
			r'$M_\mathrm{aperture}^\mathrm{(DM)}$\quad[$M_\odot$]',
			r'$M_\mathrm{aperture}^\mathrm{(stars)}$\quad[$M_\odot$]',
			r'$\theta (\mathbf{L},\mathrm{\widehat{CoP}},\mathbf{v_{pec}})$\quad[degrees]',
			r'$\theta (\mathbf{L^\mathrm{(gas)}},\mathrm{\widehat{CoP}},\mathbf{L^\mathrm{(stars)}})$\quad[degrees]$',
			r'$\theta (\mathbf{L^\mathrm{(DM)}},\mathrm{\widehat{CoP}},\mathbf{L^\mathrm{(stars)}})$\quad[degrees]$',
			r'$\theta (\mathbf{v_\mathrm{pec}^\mathrm{(gas)}},\mathrm{\widehat{CoP}},\mathbf{v_\mathrm{pec}^\mathrm{(gas)}})$\quad[degrees]$',
			r'$\mathbf{v_\mathrm{pec}}$\quad[km/s]',
			r'$\mathbf{v_\mathrm{pec}^\mathrm{(gas)}}$\quad[km/s]',
			r'$\mathbf{v_\mathrm{pec}^\mathrm{(DM)}}$\quad[km/s]',
			r'$\mathbf{v_\mathrm{pec}^\mathrm{(stars)}}$\quad[km/s]',
			r'$\mathbf{L}$\quad[$M_\odot$ Mpc km/s]',
			r'$\mathbf{L^\mathrm{(gas)}}$\quad[$M_\odot$ Mpc km/s]',
			r'$\mathbf{L^\mathrm{(DM)}}$\quad[$M_\odot$ Mpc km/s]',
			r'$\mathbf{L^\mathrm{(stars)}}$\quad[$M_\odot$ Mpc km/s]',
			r'Kinetic energy\quad[J]',
			r'Kinetic energy (gas)\quad[J]',
			r'Kinetic energy (DM)\quad[J]',
			r'Thermal energy\quad[J]',
			r'$M_\mathrm{sub}$\quad[$M_\odot$]',
			r'Dynamical merging index',
			r'Thermoynamic merging index',
			r'Substructure fraction'
	]

	def __init__(self, simulation_name:str = None, aperture_id: int = 10) -> None:
		warnings.filterwarnings("ignore")
		self.set_aperture(aperture_id)
		self.set_simulation(simulation_name)

	def set_aperture(self, aperture_id: int) -> None:
		self.aperture_id = aperture_id
		aperture_id_str = f'Aperture {self.aperture_id}'
		print(f"{aperture_id_str:^100s}\n")

	def set_simulation(self, simulation_name: str = None) -> None:
		self.simulation = Simulation(simulation_name=simulation_name)
		self.path = os.path.join(self.simulation.pathSave, self.simulation.simulation_name, 'rotvel_correlation')
		self.filename = f"simstats_{self.simulation.simulation_name}.hdf5"
		print(f"{self.simulation.simulation:=^100s}")
		if not os.path.exists(self.path):
			os.makedirs(self.path)

	def get_apertures(self, cluster: Cluster):
		read = pull.FOFRead(cluster)
		return read.pull_apertures()

	def get_matterLambda_equality_z(self) -> float:
		cluster = Cluster(simulation_name=self.simulation.simulation_name, clusterID=0, redshift='z000p000')
		omega_lambda = cluster.OmegaLambda
		omega_matter = cluster.Omega0
		return (omega_matter/(2*omega_lambda))**(-1/3) -1

	def h5store(self, filename: str, df: pd.DataFrame, key: str = 'mydata') -> None:
		with pd.HDFStore(filename) as store:
			store.put(key, df)

	def clear_file(self) -> None:
		if os.path.isfile(os.path.join(self.path, self.filename)):
			print(f"[+] Removing file...\n[+]\tPath: {self.path}\n[+]\tFile: {self.filename}")
			os.remove(os.path.join(self.path, self.filename))
			print(f"[+]\tRemoved.")

	def make_metadata(self) -> None:
		metadata = {
				'Simulation'              : self.simulation.simulation,
				'Number of clusters'      : self.simulation.totalClusters,
				'Redshift bounds'         : '0 - 1.8',
				'Sample completeness'     : np.sum(self.simulation.sample_completeness) / np.product(self.simulation.sample_completeness.shape),
				'z mL equality'           : self.get_matterLambda_equality_z(),
				'Cluster centre reference': 'Centre of potential',
				'Pipeline stage'          : r'Gadget3 - \texttt{SUBFIND} - FoFanalyser - \textbf{Simstats}',
				'Columns/labels'          : str(dict(zip(self.cols, self.labels_tex)))
		}
		print('[+] Generating attributes...')
		with h5py.File(os.path.join(self.path, self.filename), 'w') as master_file:
			for key, text in zip(metadata.keys(), metadata.values()):
				master_file.attrs.create(key, text)
		if os.path.isfile(os.path.join(self.path, self.filename)):
			print(f"[+] Saved\n[+]\tPath: {self.path}\n[+]\tFile: {self.filename}")

	def is_metadata(self) -> bool:
		if os.path.isfile(os.path.join(self.path, self.filename)):
			with h5py.File(os.path.join(self.path, self.filename), 'r') as master_file:
				return master_file.attrs is not None
		else:
			return False

	def make_simstats(self, save2hdf5: bool = True) -> Union[pd.DataFrame, None]:
		if not self.is_metadata():
			print('[+] Metadata file not found.')
			self.make_metadata()
		df = pd.DataFrame(columns=self.cols)
		iterator = itertools.product(self.simulation.clusterIDAllowed[:1], self.simulation.redshiftAllowed)
		print(f"{'':<30s} {' process ID ':^25s} | {' halo ID ':^15s} | {' halo redshift ':^20s}\n")
		for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
			if self.simulation.sample_completeness[halo_id, self.simulation.redshiftAllowed.index(halo_z)]:
				print(f"{'Processing...':<30s} {process_n:^25d} | {halo_id:^15d} | {halo_z:^20s}")
				cluster = Cluster(simulation_name=self.simulation.simulation_name,
								  clusterID=halo_id,
								  redshift=halo_z)
				read = pull.FOFRead(cluster)
				df = df.append({
					'cluster_id'     : cluster.clusterID,
					'redshift_float' : cluster.z,
					'R_2500_crit'  : cluster.r2500,
					'R_500_crit'   : cluster.r500,
					'R_200_crit'   : cluster.r200,
					'R_aperture'   : cluster.generate_apertures()[self.aperture_id],
					'M_2500_crit'  : cluster.mass_units(cluster.M2500, unit_system='astro'),
					'M_500_crit'   : cluster.mass_units(cluster.M500, unit_system='astro'),
					'M_200_crit'   : cluster.mass_units(cluster.M200, unit_system='astro'),
					'M_aperture_T' : cluster.mass_units(read.pull_mass_aperture('Total_mass')[self.aperture_id], unit_system='astro'),
					'M_aperture_0' : cluster.mass_units(read.pull_mass_aperture('ParType0_mass')[self.aperture_id], unit_system='astro'),
					'M_aperture_1' : cluster.mass_units(read.pull_mass_aperture('ParType1_mass')[self.aperture_id], unit_system='astro'),
					'M_aperture_4' : cluster.mass_units(read.pull_mass_aperture('ParType4_mass')[self.aperture_id], unit_system='astro'),
					'rotTvelT' : read.pull_rot_vel_angle_between('Total_angmom', 'Total_ZMF')[self.aperture_id],
					'rot0rot4' : read.pull_rot_vel_angle_between('ParType0_angmom', 'ParType4_angmom')[self.aperture_id],
					'rot1rot4' : read.pull_rot_vel_angle_between('ParType1_angmom', 'ParType4_angmom')[self.aperture_id],
					'vel0vel1' : read.pull_rot_vel_angle_between('ParType0_ZMF', 'ParType1_ZMF')[self.aperture_id],
					'peculiar_velocity_T_magnitude' : read.pull_peculiar_velocity_magnitude('Total_ZMF')[self.aperture_id],
					'peculiar_velocity_0_magnitude' : read.pull_peculiar_velocity_magnitude('ParType0_ZMF')[self.aperture_id],
					'peculiar_velocity_1_magnitude' : read.pull_peculiar_velocity_magnitude('ParType1_ZMF')[self.aperture_id],
					'peculiar_velocity_4_magnitude' : read.pull_peculiar_velocity_magnitude('ParType4_ZMF')[self.aperture_id],
					'angular_momentum_T_magnitude' : cluster.momentum_units(read.pull_angular_momentum_magnitude('Total_angmom')[self.aperture_id], unit_system='astro'),
					'angular_momentum_0_magnitude' : cluster.momentum_units(read.pull_angular_momentum_magnitude('ParType0_angmom')[self.aperture_id], unit_system='astro'),
					'angular_momentum_1_magnitude' : cluster.momentum_units(read.pull_angular_momentum_magnitude('ParType1_angmom')[self.aperture_id], unit_system='astro'),
					'angular_momentum_4_magnitude' : cluster.momentum_units(read.pull_angular_momentum_magnitude('ParType4_angmom')[self.aperture_id], unit_system='astro'),
					'kinetic_energy_T'    : cluster.energy_units(read.pull_kinetic_energy('Total_kin_energy')[self.aperture_id], unit_system='SI'),
					'kinetic_energy_0'    : cluster.energy_units(read.pull_kinetic_energy('ParType0_kin_energy')[self.aperture_id], unit_system='SI'),
					'kinetic_energy_1'    : cluster.energy_units(read.pull_kinetic_energy('ParType1_kin_energy')[self.aperture_id], unit_system='SI'),
					'thermal_energy'      : cluster.energy_units(read.pull_thermal_energy('Total_th_energy')[self.aperture_id], unit_system='SI'),
					'substructure_mass_T' : cluster.mass_units(read.pull_substructure_mass('Total_substructure_mass')[self.aperture_id], unit_system='astro'),
					'dynamical_merging_index_T'     : read.pull_dynamical_merging_index('Total_dyn_mergindex')[self.aperture_id],
					'thermodynamic_merging_index_T' : read.pull_thermodynamic_merging_index('Total_therm_mergindex')[self.aperture_id],
					'substructure_fraction_T'       : read.pull_substructure_merging_index('Total_substructure_fraction')[self.aperture_id]
				}, ignore_index=True)
			else:
				print(f"{'Skip - sample_completeness':<30s} {process_n:^25d} | {halo_id:^15d} | {halo_z:^20s}")

		print(df.info())
		if save2hdf5:
			self.h5store(os.path.join(self.path, self.filename), df, key=f'aperture{self.aperture_id}')
			if os.path.isfile(os.path.join(self.path, self.filename)):
				print(f"[+] Saved\n[+]\tPath: {self.path}\n[+]\tFile: {self.filename}")
		else:
			return df

	def read_metadata(self) -> Dict[str, Union[int, float, str, Dict[str, str]]]:
		with h5py.File(os.path.join(self.path, self.filename), 'r') as master_file:
			metadata = dict(zip(master_file.attrs.keys(), master_file.attrs.values()))
			metadata['Columns/labels'] = eval(metadata['Columns/labels'])
			return metadata

	def read_simstats(self) -> pd.DataFrame:
		with pd.HDFStore(os.path.join(self.path, self.filename)) as store:
			return store[f'aperture{self.aperture_id}']


if __name__ == '__main__':

	simstats = Simstats(simulation_name='celr_b', aperture_id=10)
	simstats.clear_file()
	simstats.make_simstats(save2hdf5=True)
	stats_out = simstats.read_simstats()
	print(stats_out.query('cluster_id == 0 and redshift_float < 0.1')['redshift_float'])
	print(simstats.read_metadata())