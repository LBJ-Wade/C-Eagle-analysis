import sys
import os.path
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from save import dict2hdf as write
from rotvel_correlation.alignment import group_alignment


def data2cluster(data: dict) -> Cluster:
	pass

def alignment_report(cluster: Cluster):
	apertures = cluster.generate_apertures()
	master_dict = {}
	for i, r_a in enumerate(apertures):
		halo_output = {
				**cluster.group_fofinfo(aperture_radius=r_a),
				**cluster.group_dynamics(aperture_radius=r_a),
				**cluster.group_morphology(aperture_radius=r_a)
		}
		alignment_dict = group_alignment(halo_output)
		halo_output = {
				**halo_output,
				**alignment_dict
		}
		master_dict[f'aperture{i:02d}'] = halo_output
		del halo_output, alignment_dict

	if not os.path.exists(os.path.join(cluster.pathSave, 'alignment_project')):
		os.makedirs(os.path.join(cluster.pathSave, 'alignment_project'))
	pathFile = os.path.join(cluster.pathSave, 'alignment_project', f"{cluster.redshift}")
	if not os.path.exists(pathFile):
		os.makedirs(pathFile)
	write.save_dict_to_hdf5(master_dict, os.path.join(pathFile, f"halo_{cluster.clusterID}.hdf5"))
	del cluster