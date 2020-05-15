import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import slack
import warnings
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2

# exec(open(os.path.abspath(os.path.join(
# 		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
warnings.filterwarnings("ignore")
from import_toolkit.cluster import Cluster

def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None,
                 fill=False, fill_kwargs=None, data_out=False, cov=None, mass_level=0.68):
    '''
        An easy to use function for plotting ellipses in Python 2.7!

        The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
        It can take a covariance matrix and plot contours from it.

        semimaj : float
            length of semimajor axis (always taken to be some phi (-90<phi<90 deg) from positive x-axis!)

        semimin : float
            length of semiminor axis

        phi : float
            angle in radians of semimajor axis above positive x axis

        x_cent : float
            X coordinate center

        y_cent : float
            Y coordinate center

        theta_num : int
            Number of points to sample along ellipse from 0-2pi

        ax : matplotlib axis property
            A pre-created matplotlib axis

        plot_kwargs : dictionary
            matplotlib.plot() keyword arguments

        fill : bool
            A flag to fill the inside of the ellipse

        fill_kwargs : dictionary
            Keyword arguments for matplotlib.fill()

        data_out : bool
            A flag to return the ellipse samples without plotting

        cov : ndarray of shape (2,2)
            A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi

        mass_level : float
            if supplied cov, mass_level is the contour defining fractional probability mass enclosed
            for example: mass_level = 0.68 is the standard 68% mass

    '''
    # Get Ellipse Properties from cov matrix
    if cov is not None:
        eig_vec, eig_val, u = np.linalg.svd(cov)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semimaj = np.sqrt(eig_val[0])
        semimin = np.sqrt(eig_val[1])
        if mass_level is None:
            multiplier = np.sqrt(2.279)
        else:
            distances = np.linspace(0, 20, 20001)
            chi2_cdf = chi2.cdf(distances, df=2)
            multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
        semimaj *= multiplier
        semimin *= multiplier
        phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0, 2 * np.pi, theta_num)
    r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.array([x, y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    data[0] += x_cent
    data[1] += y_cent

    # Output data?
    if data_out == True:
        return data

    # Plot!
    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots()

    if plot_kwargs is None:
        ax.plot(data[0], data[1], color='b', linestyle='-')
    else:
        ax.plot(data[0], data[1], **plot_kwargs)

    if fill == True:
        ax.fill(data[0], data[1], **fill_kwargs)

    if return_fig == True:
        return fig


filepath = "/local/scratch/altamura/analysis_results/"
filename = f"bahamas-clustermap-5r200.jpg"
data_required = {'partType0': ['groupnumber', 'mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
                 'partType1': ['groupnumber', 'mass', 'coordinates', 'velocity'],
                 'partType4': ['groupnumber', 'mass', 'coordinates', 'velocity']}
cluster = Cluster(simulation_name='bahamas',
                  clusterID=0,
                  redshift='z003p000',
                  comovingframe=False,
                  fastbrowsing=False,
                  requires=data_required)

coords = cluster.partType0_coordinates
x0, y0, z0 = cluster.centre_of_potential
morphology = cluster.group_morphology(aperture_radius=cluster.r200)
eigenvalues = morphology['eigenvalues'][1]
eigenvectors = morphology['eigenvectors'][1].reshape((3,3))

# Sort eigenvalues from largest to smallest
eigenvalues  = [x for x,_ in sorted(zip(eigenvalues,eigenvectors))][::-1]
eigenvectors = [x for _,x in sorted(zip(eigenvalues,eigenvectors))][::-1]

a_val, b_val, c_val = np.sqrt(eigenvalues)
a_vec, b_vec, c_vec = eigenvectors
x = coords[:,0]
y = coords[:,1]
del coords

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel(r'$y\ $ [Mpc]')
ax.set_ylabel(r'$y\ $ [Mpc]')
ax.scatter(x,y, marker=',', c='k', s=1, alpha=0.07)
ax.scatter([x0], [y0], marker='*', c='r', s=40, alpha=1)
ax.plot([x0, x0+a_vec[0]], [y0, y0+a_vec[1]], marker=None, c='lime', lw=1, alpha=1)
ax.plot([x0, x0+b_vec[0]], [y0, y0+b_vec[1]], marker=None, c='lime', lw=1, alpha=1)
plot_ellipse(x_cent = x0,
             y_cent = y0,
             semimaj=a_val,
             semimin=b_val,
             phi=cluster.angle_between_vectors(a_vec[:2], [0,1]),
             ax=ax,
             plot_kwargs={'color':'r','linestyle':'-','linewidth':3,'alpha':0.8})

items_labels = r"""POINT PARTICLE MAP
Cluster {:s} {:d}
$z$ = {:.2f}
$R_{{500\ true}}$ = {:.2f} Mpc
Triaxiality = {:.2f}
Circularity = {:.2f}""".format(cluster.simulation,
                               cluster.clusterID,
                               cluster.z,
                               cluster.r500,
                               morphology['triaxiality'][0],
                               morphology['circularity'][0])
print(items_labels)
ax.text(0.03, 0.97, items_labels,
          horizontalalignment='left',
          verticalalignment='top',
          transform=ax.transAxes,
          size=15)

plt.savefig(filepath+filename)

# Send files to Slack: init slack client with access token
print(f"[+] Forwarding {filename} to the `#personal` Slack channel...")
slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
client = slack.WebClient(token=slack_token)
response = client.files_upload(
        file=f"{filepath+filename}",
        initial_comment=f"This file was sent upon completion of the plot factory pipeline.\nAttachments: {filename}",
        channels='#personal'
)