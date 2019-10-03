import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import kernel_convolver as kernconv
from skimage.feature import blob_log

def substructure_identification(map, rfov, r200, plotmap = False, axes = None):
    if map.shape[0] != map.shape[1]:
        print("Map is not square!")
        exit(0)

    nbins = np.size(map, 0)+1
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    dx, dy = x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]

    # Grayscale image
    map_array = np.reshape(np.log10(np.abs(map)), (1, (nbins-1)  ** 2))
    img_par_scale = 0.75 * np.max(map_array)
    img_par_offset = np.percentile(map_array, 75) / img_par_scale

    aux_map = img_par_offset - np.log10(np.abs(map)) / img_par_scale

    aux_map[aux_map == -np.inf] = 0
    aux_map[aux_map < 0] = 0
    aux_map[aux_map > 1] = 1

    _, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type='gauss')

    # Blob identification
    test = fwhm / (60 * dx)

    min_sigma = test * 1
    max_sigma = test * 10
    num_sigma = 80
    threshold = 0.05

    blobs_log = blob_log(aux_map, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    # Take raw output from the blob log algorithm
    bly = blobs_log[:, 0]
    blx = blobs_log[:, 1]
    blr = blobs_log[:, 2]

    # Create data storage for the blobl markers and convert to plot axis coordinate system
    blob_catalog_dict = {'I': 0,
                         'X': 0,
                         'Y': 0,
                         'R': 0}

    blob_catalog_dict['I'] = np.arange(len(blx))
    blob_catalog_dict['X'] = (blx - nbins / 2 + 1) / nbins * rfov * r200 * 2
    blob_catalog_dict['Y'] = (bly - nbins / 2 + 1) / nbins * rfov * r200 * 2
    blob_catalog_dict['R'] = blr / nbins * rfov * r200 * 2

    if plotmap and axes is not None:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        Cx, Cy = np.meshgrid((x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2)
        img = axes.pcolor(Cx, Cy, aux_map, cmap='gray', norm=norm)

        # Render elements in plots
        axes.set_aspect('equal')
        axes.set_xlim(-rfov * r200, rfov * r200)
        axes.set_ylim(-rfov * r200, rfov * r200)
        axes.axis('off')

        """ 
        circle = mpatches.Circle((0,0), radius=r200, color='cyan', linestyle='--', alpha=1, fill=False)
        axes.add_patch(circle)
        circle = mpatches.Circle((0,0), radius=5*r200, color='cyan', alpha=1, fill=False)
        axes.add_patch(circle)
        """

        # Colorbar adjustments
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        ax2_divider = make_axes_locatable(axes)
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbarlabel = r'$y^{\mathrm{(scaled)}}_{\mathrm{kSZ}}$'
        cbar.set_label(cbarlabel, labelpad=-70)
        cax2.xaxis.set_ticks_position("top")

    return blob_catalog_dict