import sys
import os
from typing import Dict, List
import warnings
import itertools
import h5py
import numpy as np
import pandas as pd
import scipy.stats as st
from multiprocessing.dummy import Pool as ThreadPool

basepath = '/local/scratch/altamura/analysis_results/alignment_project'

def redshift_str2num(z: str) -> float:
	"""
	Converts the redshift of the snapshot from text to numerical,
	in a format compatible with the file names.
	E.g. float z = 2.16 <--- str z = 'z002p160'.
	"""
	z = z.strip('z').replace('p', '.')
	return round(float(z), 3)

def pull_halo_output(h5file, clusterID, apertureID, dataset):
	"""
	Function to extract a dataset from a Bahamas snapshot output.
	:param h5file: The h5py File object to extract the data from
	:param clusterID: The number of cluster in the order imposed by FoF
	:param apertureID: int(0-22) The index of the spherical aperture centred on the CoP
	:param dataset: The name of the dataset to pull
	:return: None if the cluster does not exist in the file, the dataset as np.ndarray if it exists
	"""
	if f'halo_{clusterID:05d}' not in h5file:
		warnings.warn(f"[-] Cluster {clusterID} not found in snap output.")
		return None
	else:
		return h5file[f'halo_{clusterID:05d}/aperture{apertureID:02d}/{dataset}'][...]

def read_snap_output(redshift: str, apertureID: int = None, dataset: str = None) -> np.ndarray:
	"""
	Function to collect datasets from all clusters at given redshift and aperture.
	The function is a wrapper around `pull_halo_output`, called within the multiprocessing/threading
	modules. It multithreads the I/O function without the use of MPI for performance.
	:param redshift: The redshift as a string in EAGLE format
	:param apertureID: int(0-22) The index of the spherical aperture centred on the CoP
	:param dataset: The name of the dataset to pull
	:return: The merged dataset selected taken from all snapshot clusters.
	"""
	snapname = f'bahamas_hyd_alignment_{redshift}.hdf5'
	h5file = h5py.File(os.path.join(basepath, snapname), 'r')
	last_halo_key = list(h5file.keys())[-1]
	last_halo_id = int(last_halo_key[-5:])
	clusterIDs = list(range(last_halo_id))

	results = []
	for i in clusterIDs:
		results.append(pull_halo_output(h5file, i, apertureID, dataset))

	# # Make the Pool of workers
	# pool = ThreadPool(12)
	# results = pool.starmap(
	# 		pull_halo_output,
	# 		zip(
	# 				itertools.repeat(h5file),
	# 				clusterIDs,
	# 				itertools.repeat(apertureID),
	# 				itertools.repeat(dataset)
	# 		)
	# )
	#
	# # Close the pool and wait for the work to finish
	# h5file.close()
	# pool.close()
	# pool.join()
	results = [x for x in results if x]
	results = np.asarray(results)
	return results

def output_as_dict(*args, **kwargs) -> Dict[str, np.ndarray]:
	snap_out = {}
	for ds in datasets_names:
		snap_out[ds] = read_snap_output(*args, dataset=ds, **kwargs)
	return snap_out

def output_as_pandas(*args, **kwargs) -> pd.DataFrame:
	snap_dict = output_as_pandas(*args, **kwargs)
	snap_pd = pd.DataFrame(data=snap_dict, columns=snap_dict.keys())
	del snap_dict
	return snap_pd

def bayesian_blocks(t: np.ndarray) -> np.ndarray:
	"""
	Bayesian Blocks Implementation
	By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
	:param t: ndarray, length N data to be histogrammed
	:return: ndarray array containing the (N+1) bin edges
	"""

	# copy and sort the array
	t = np.sort(t)
	N = len(t)

	# create length-(N + 1) array of cell edges
	edges = np.concatenate([t[:1],
	                        0.5 * (t[1:] + t[:-1]),
	                        t[-1:]])
	block_length = t[-1] - edges

	# arrays needed for the iteration
	nn_vec = np.ones(N)
	best = np.zeros(N, dtype=float)
	last = np.zeros(N, dtype=int)

	#-----------------------------------------------------------------
	# Start with first data cell; add one cell at each iteration
	#-----------------------------------------------------------------
	for K in range(N):
		# Compute the width and count of the final bin for all possible
		# locations of the K^th changepoint
		width = block_length[:K + 1] - block_length[K + 1]
		count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

		# evaluate fitness function for these possibilities
		fit_vec = count_vec * (np.log(count_vec) - np.log(width))
		fit_vec -= 4  # 4 comes from the prior on the number of changepoints
		fit_vec[1:] += best[:K]

		# find the max of the fitness: this is the K^th changepoint
		i_max = np.argmax(fit_vec)
		last[K] = i_max
		best[K] = fit_vec[i_max]

	#-----------------------------------------------------------------
	# Recover changepoints by iteratively peeling off the last block
	#-----------------------------------------------------------------
	change_points =  np.zeros(N, dtype=int)
	i_cp = N
	ind = N
	while True:
		i_cp -= 1
		change_points[i_cp] = ind
		if ind == 0:
			break
		ind = last[ind - 1]
	change_points = change_points[i_cp:]

	return edges[change_points]

def shimazaki_shinomoto(data: np.ndarray) -> np.ndarray:
	"""
	Based on : Shimazaki H. and Shinomoto S., A method for selecting the bin size
	of a time histogram Neural Computation (2007)
	:param x:
	:return:
	"""
	data_max = max(data)  # lower end of data
	data_min = min(data)  # upper end of data
	n_min = 2  # Minimum number of bins Ideal value = 2
	n_max = 200  # Maximum number of bins  Ideal value =200
	n_shift = 30  # number of shifts Ideal value = 30
	N = np.array(range(n_min, n_max))
	D = float(data_max - data_min) / N  # Bin width vector
	Cs = np.zeros((len(D), n_shift))  # Cost function vector
	# Computation of the cost function
	for i in range(np.size(N)):
		shift = np.linspace(0, D[i], n_shift)
		for j in range(n_shift):
			edges = np.linspace(data_min + shift[j] - D[i] / 2, data_max + shift[j] - D[i] / 2, N[i] + 1)  # shift the Bin edges
			binindex = np.digitize(data, edges)  # Find binindex of each data point
			ki = np.bincount(binindex)[1:N[i] + 1]  # Find number of points in each bin
			k = np.mean(ki)  # Mean of event count
			v = sum((ki - k) ** 2) / N[i]  # Variance of event count
			Cs[i, j] += (2 * k - v) / ((D[i]) ** 2)  # The cost Function
	C = Cs.mean(1)

	# Optimal Bin Size Selection
	loc = np.argwhere(Cs == Cs.min())[0]
	cmin = C.min()
	idx = np.where(C == cmin)
	idx = idx[0][0]
	optD = D[idx]
	edges = np.linspace(data_min + shift[loc[1]] - D[idx] / 2, data_max + shift[loc[1]] - D[idx] / 2, N[idx] + 1)
	return edges

def freedman_diaconis(x: np.ndarray) -> np.ndarray:
    """
    The binwidth is proportional to the interquartile range (IQR) and inversely proportional to cube root of a.size.
    Can be too conservative for small datasets, but is quite good for large datasets. The IQR is very robust to
    outliers.

    :param x: np.ndarray
        The 1-dimensional x-data to bin.
    :return: np.ndarray
        The bins edges computed using the FD method.
    """
    return np.histogram_bin_edges(x, bins='fd')

def equal_number(x: np.ndarray) -> np.ndarray:
    """
    Takes the number of bins computed using the FD method, but then selects the bin edges splitting
    the dataset in bins with equal number of data-points.

    :param x: np.ndarray
        The 1-dimensional x-data to bin.
    :return: np.ndarray
        The bins edges computed using the equal-N method.
    """
    nbin = 20
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def kde_2d(x: np.ndarray, y: np.ndarray, axscales: List[str] = None, gridbins: int = None) -> tuple:
    """
    Function to compute the 2D kernel density eatimate of a 2D dataset.
    It accepts linear and logarithmic scales on both axes and rescales the kde accordingly.
    :param x: The array for x-values
    :param y: The array for y-values
    :param axscales: A list with 2 string entries with the scales of the axes
    :return: Tuple with the x and y gridmesh for the KDE and the z-values for contours
    """
    if not gridbins:
        gridbins = 101
    if not axscales:
        axscales = ['linear', 'linear']

    if axscales[0] == 'linear':
        x_space = np.linspace(np.min(x), np.max(x), gridbins)
    elif axscales[0] == 'log':
        x_space = np.linspace(np.log10(np.min(x)), np.log10(np.max(x)), gridbins)
    if axscales[1] == 'linear':
        y_space = np.linspace(np.min(y), np.max(y), gridbins)
    elif axscales[1] == 'log':
        y_space = np.linspace(np.log10(np.min(y)), np.log10(np.max(y)), gridbins)

    xx, yy = np.meshgrid(x_space, y_space)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x if axscales[0] == 'linear' else np.log10(x), y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    if axscales[0] == 'linear':
        return xx, yy, f
    else:
        return 10**xx, yy, f

def medians_2d(x: np.ndarray, y: np.ndarray, axscales: List[str] = None, binning_method: str = None) -> dict:
	"""

	:param x: The array for x-values
	:param y: The array for y-values
	:param axscales: A list with 2 string entries with the scales of the axes
	:param binning_method:
	:return:
	"""
	if not axscales:
		axscales = ['linear', 'linear']

	if binning_method == 'bayesian':
		x_binning = bayesian_blocks
	elif binning_method == 'freedman':
		x_binning = freedman_diaconis
	elif binning_method == 'equalnumber':
		x_binning = equal_number
	elif binning_method == 'shimazaki_shinomoto':
		x_binning = shimazaki_shinomoto
	elif binning_method == None:
		x_binning = bayesian_blocks

	if axscales[0] == 'linear':
		x_bin_stats = x_binning(x)
	elif axscales[0] == 'log':
		x_bin_stats = 10 ** x_binning(np.log10(x))

	# Remove duplicate bin edges
	# x_bin_stats = np.sort(list(set(x_bin_stats)))

	median_y, edges, _ = st.binned_statistic(x, y, statistic='median', bins=x_bin_stats)
	percent84_y, _, _ = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 84), bins=x_bin_stats)
	percent16_y, _, _ = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 16), bins=x_bin_stats)
	count_y, _, _ = st.binned_statistic(x, y, statistic='count', bins=x_bin_stats)
	std_y, _, _ = st.binned_statistic(x, y, statistic='std', bins=x_bin_stats)
	median_x = edges[: -1] + np.diff(edges) / 2

	median_stats = {}
	median_stats['median_x'] = median_x
	median_stats['median_y'] = median_y
	median_stats['percent84_y'] = percent84_y
	median_stats['percent16_y'] = percent16_y
	median_stats['count_y'] = count_y
	median_stats['err_y'] = std_y/np.sqrt(count_y)
	del median_y, percent84_y, percent16_y, count_y, std_y, median_x
	return median_stats

def get_label_between(label: str, limiters: List[str] = ['(', ')']) -> str:
	s = label.split(limiters[0], 1)[1]
	s = s.split(limiters[1], 1)[0]
	return s


"""
clevels = ax.contour(xmesh,ymesh,H.T,lw=.9,cmap='winter')#,zorder=90)

# Identify points within contours
p = clevels.collections[0].get_paths()
inside = np.full_like(x,False,dtype=bool)
for level in p:
    inside |= level.contains_points(zip(*(x,y)))

ax.plot(x[~inside],y[~inside],'kx')
"""


output_datasets = [
    'N_particles',
    'NumOfSubhalos',
    'Omega0',
    'OmegaBaryon',
    'OmegaLambda',
    'a_l',
    'a_v',
    'a_w',
    'angular_momentum',
    'angular_velocity',
    'aperture_mass',
    'b_l',
    'b_v',
    'b_w',
    'c_l',
    'c_v',
    'c_w',
    'centre_of_mass',
    'centre_of_potential',
    'circular_velocity',
    'dynamical_merging_index',
    'eigenvalues',
    'eigenvectors',
    'elongation',
    'hubble_param',
    'inertia_tensor',
    'kinetic_energy',
    'l_w',
    'm200',
    'm2500',
    'm500',
    'mfof',
    'r200',
    'r2500',
    'r500',
    'r_aperture',
    'redshift',
    'specific_angular_momentum',
    'sphericity',
    'spin_parameter',
    'substructure_fraction',
    'substructure_mass',
    'thermal_energy',
    'thermodynamic_merging_index',
    'triaxiality',
    'v_l',
    'v_w',
    'zero_momentum_frame'
]

output_labels = [
    r'$N_\mathrm{particles}$',
    r'$N_{sub}$',
    r'$\Omega_0$',
    r'$\Omega_b$',
    r'$\Omega_\Lambda$',
    r'$\theta(\mathbf{a}_\mathcal{I}, \mathbf{J})\quad$[degrees]',
    r'$\theta(\mathbf{a}_\mathcal{I}, \mathbf{v}_p)\quad$[degrees]',
    r'$\theta(\mathbf{a}_\mathcal{I}, \mathbf{\omega})\quad$[degrees]',
    r'$\mathbf{J}$',
    r'$\mathbf{\omega}$',
    r'$M_\mathrm{aperture}\quad$[M$_\odot$]',
    r'$\theta(\mathbf{b}_\mathcal{I}, \mathbf{J})\quad$[degrees]',
    r'$\theta(\mathbf{b}_\mathcal{I}, \mathbf{v}_p)\quad$[degrees]',
    r'$\theta(\mathbf{b}_\mathcal{I}, \mathbf{\omega})\quad$[degrees]',
    r'$\theta(\mathbf{c}_\mathcal{I}, \mathbf{J})\quad$[degrees]',
    r'$\theta(\mathbf{c}_\mathcal{I}, \mathbf{v}_p)\quad$[degrees]',
    r'$\theta(\mathbf{c}_\mathcal{I}, \mathbf{\omega})\quad$[degrees]',
    r'Centre of mass',
    r'Centre of potential',
    r'Circular velocity$\quad$[km\,s$^{-1}$]',
    r'Dynamical merging index',
    r'Eigenvalues',
    r'Eigenvectors',
    r'Elongation',
    r'h',
    r'$inertia_tensor$',
    r'$kinetic_energy$',
    r'$\theta(\mathbf{J}, \mathbf{\omega})\quad$[degrees]',
    r'$M_{200}\quad$[M$_\odot$]',
    r'$M_{2500}\quad$[M$_\odot$]',
    r'$M_{500}\quad$[M$_\odot$]',
    r'$M_{FoF}\quad$[M$_\odot$]',
    r'$R_{200}\quad$[Mpc]',
    r'$R_{2500}\quad$[Mpc]',
    r'$R_{500}\quad$[Mpc]',
    r'$R_\mathrm{aperture}\quad$[Mpc]',
    r'Redshift',
    r'$J/M$',
    r'Sphericity',
    r'Spin parameter',
    r'$M_{sub}/M$',
    r'$M_{sub}\quad$[M$_\odot$]',
    r'Thermal energy',
    r'Thermodynamic merging index',
    r'Triaxiality',
    r'$\theta(\mathbf{v}_p, \mathbf{J})\quad$[degrees]',
    r'$\theta(\mathbf{v}_p, \mathbf{\omega})\quad$[degrees]',
    r'$\mathbf{v}_{p}\quad$[km\,s$^{-1}$]'
]

datasets_names = dict(zip(output_datasets, output_labels))

aperture_labels = [
    '80 kpc',
    '100 kpc',
    '200 kpc',
    '400 kpc',
    '0.1 $R_{500}$',
    '$R_{2500}$',
    '1.5 $R_{2500}$',
    '$R_{500}$',
    '$R_{200}$',
    '1.12 $R_{200}$',
    '1.26 $R_{200}$',
    '1.41 $R_{200}$',
    '1.58 $R_{200}$',
    '1.78 $R_{200}$',
    '1.99 $R_{200}$',
    '2.24 $R_{200}$',
    '2.51 $R_{200}$',
    '2.81 $R_{200}$',
    '3.16 $R_{200}$',
    '3.54 $R_{200}$',
    '3.97 $R_{200}$',
    '4.46 $R_{200}$',
    '5.00 $R_{200}$'
]

partType_labels = {
		0 : 'total',
		1 : 'gas',
		2 : 'CDM',
		3 : 'stars'
}