import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import subhalo_marker as sm
import clusters_retriever as extract
import matplotlib.pyplot as plt
import warnings
from matplotlib import rc

def compare(cat_subfind, cat_blobs):
    """INPUT REQUIREMENTS:
        cat_subfind is the catalog cointaing coordinates of subfind subhalos, needs X and Y values
        cat_blobs is the catalog of blobs and needs to have X, Y positions and radius R
        positions must be passed as plotted (i.e. in angular scale - to prevent repetitive calculations)

        Input is not modified by correspondence between indices of blobs and subs within is preserved
    """
    # Check input requirements
    if not ('X' in cat_blobs and 'Y' in cat_blobs and 'R' in cat_blobs and 'I' in cat_blobs and 'X' in cat_subfind and 'Y' in cat_subfind and 'I' in cat_subfind):
        print('Error: Input catalog has wrong format')
        exit(1)

    # Every blob has an array of markers within associated with it
    subs_within = np.zeros(len(cat_blobs['I']))

    for i in range(len(cat_blobs['I'])):
        gather_markers = []
        for j in range(len(cat_subfind['I'])):
            if (cat_blobs['X'][i] - cat_subfind['X'][j]) ** 2 + (
                    cat_blobs['Y'][i] - cat_subfind['Y'][j]) ** 2 < (cat_blobs['R'][i]) ** 2:
                gather_markers.append(int(cat_subfind['I'][j]))

        # Assign values
        if len(gather_markers) > 0:
            subs_within[i] = int(np.min(gather_markers))
        elif subs_within[i] == 0.:
            subs_within[i] = np.nan

    return subs_within

def render_figure(axes, cat_blobs, cat_subfind, subs_within, annotate=True):
    legends = [r'$\mathrm{\texttt{SUBFIND}\ catalogue}$',
               r'$\mathrm{LoG\ detection\ catalogue}$',
               r'$\mathrm{Matched\ subhalos}$']

    ax_label = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$']

    # ax.scatter(peak_detection['X'], peak_detection['Y'], s=(30 * peak_detection['R'])**2, marker='o', facecolors='none', edgecolors='lime', alpha=0.5, label = legends[1])
    # Vary colors and alphas of the patches
    x = cat_blobs['R']
    #alphas = (x - (np.min(x) - 0.5)) / ((np.max(x) + 1) - (np.min(x) - 0.5))
    alphas = 0.7*np.ones_like(x)

    legend_trigger = True
    for i in range(len(cat_blobs['I'])):
        # Plot blobs
        if legend_trigger:
            circle = mpatches.Circle((cat_blobs['X'][i], cat_blobs['Y'][i]), radius=cat_blobs['R'][i],
                                     color='green', alpha=alphas[i], label=legends[1])
        else:
            circle = mpatches.Circle((cat_blobs['X'][i], cat_blobs['Y'][i]), radius=cat_blobs['R'][i],
                                     color='green', alpha=alphas[i])
        axes.add_patch(circle)
        legend_trigger = False

    axes.scatter(cat_subfind['X'], cat_subfind['Y'], c='cyan', marker='.', s=30, linewidths=0.02, alpha=0.5, label=legends[0])
    
    legend_trigger = True
    for i in range(len(cat_blobs['I'])):
        # Plot matched markers
        idx = np.where(cat_subfind['I'] == subs_within[i])[0]
        if legend_trigger:
            axes.scatter(cat_subfind['X'][idx], cat_subfind['Y'][idx], color='k', marker='x', s=30, linewidths=0.002,
                               alpha=0.8, label=legends[2], zorder=5)
        else:
            axes.scatter(cat_subfind['X'][idx], cat_subfind['Y'][idx], color='k', marker='x', s=30, linewidths=0.002,
                         alpha=0.8, zorder=5)
        #for j in idx:
        #    if annotate: axes.annotate(str(cat_subfind['I'][j]), xy=(cat_subfind['X'][j], cat_subfind['Y'][j]),
        #                xytext=(-5, 5), textcoords='offset points', size=7)
        #    if legend_trigger:
        #        #circle_1 = mpatches.Circle((cat_subfind['X'][j], cat_subfind['Y'][j]), radius=0.2, color='k',
        #        #                          alpha=(0.5 + alphas[i] / 2), label=legends[2])
        #        cross_1 = axes.scatter(cat_subfind['X'][j], cat_subfind['Y'][j], color='k', marker='x', s=30, linewidths=0.02, alpha=0.8, label=legends[2])
        #    else:
        #        #circle_1 = mpatches.Circle((cat_subfind['X'][j], cat_subfind['Y'][j]), radius=0.2, color='k',
        #        #                          alpha=(0.5 + alphas[i] / 2))
        #        cross_2 = axes.scatter(cat_subfind['X'][j], cat_subfind['Y'][j], color='k', marker='x', s=30, linewidths=0.02, alpha=0.8)
        #    #axes.add_patch(circle_1)
        legend_trigger = False

    # Plot legend.
    lgnd = axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                     borderaxespad=0., numpoints=1, fontsize=16, shadow=False, fancybox=True,
                     facecolor='white', edgecolor='grey')

    # change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [60]
    lgnd.legendHandles[1]._sizes = [60]
    lgnd.legendHandles[2]._sizes = [60]
    axes.set_xlabel(ax_label[0])
    axes.set_ylabel(ax_label[1])
    axes.set_aspect('equal')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    # rc('text', usetex=True)

    # Import dictionaries
    subfind_selected = np.load('subfind_catalog_FullData.npy').item()
    peak_detection = np.load('Substructure-Identification-kSZ//blob_catalog_kSZ_halo0_z057_rfov5_nbins300.npy').item()
    sg_catalogue = np.load('subhalo_catalog.npy')


    # Get subfind catalog x'y coordinates of markers
    # Use function in subhalo_marker.py
    num_halo = 0
    redshift = 0.57
    path = extract.path_from_cluster_name(num_halo, simulation_type = 'gas')
    file = extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    subfind_xyz = sm.subhalo_marks_filtered(path, file, sg_catalogue)

    subfind_coords = {'x' : subfind_xyz[0],
                      'y' : subfind_xyz[1],
                      'z' : subfind_xyz[2],
                      'idx' : sg_catalogue}

    # Every blob has an array of markers within associated with it
    peak_detection['subs_within'] = np.zeros(len(peak_detection['I']))

    for i in range(len(peak_detection['I'])):
        gather_markers = []
        for j in range(len(subfind_selected['index_catalog'])):
            if (peak_detection['X'][i] - subfind_coords['x'][j])**2 + (peak_detection['Y'][i] - subfind_coords['y'][j])**2 < (peak_detection['R'][i])**2:
                gather_markers.append(int(subfind_selected['index_catalog'][j]))

        # Assign values
        if len(gather_markers)>0:
            peak_detection['subs_within'][i] = np.min(gather_markers)
        elif peak_detection['subs_within'][i] == 0.:
            peak_detection['subs_within'][i] = np.nan

    # PLOT FIGURE
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    legends = [r'$\mathrm{Subfind\ catalog}$',
               r'$\mathrm{Subhalo\ detection}$',
               r'$\mathrm{Matched\ [sg\ number]}$']

    ax_label = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$']

    # ax.scatter(peak_detection['X'], peak_detection['Y'], s=(30 * peak_detection['R'])**2, marker='o', facecolors='none', edgecolors='lime', alpha=0.5, label = legends[1])
    # Vary colors and alphas of the patches
    x = peak_detection['R']
    alphas = (x-(np.min(x)-0.5))/((np.max(x)+1)-(np.min(x)-0.5))

    legend_trigger = False
    for i in range(len(peak_detection['I'])):
        # Plot blobs
        if legend_trigger:
            circle = mpatches.Circle((peak_detection['X'][i], peak_detection['Y'][i]), radius=peak_detection['R'][i], color='orange', alpha=alphas[i], label = legends[1])
        else:
            circle = mpatches.Circle((peak_detection['X'][i], peak_detection['Y'][i]), radius=peak_detection['R'][i], color='orange', alpha=alphas[i])
        ax.add_patch(circle)
        legend_trigger = False

    ax.scatter(subfind_coords['x'], subfind_coords['y'], c='cyan', marker='x', s=30, linewidths=0.02, alpha=0.4, label = legends[0])

    legend_trigger = False
    for i in range(len(peak_detection['I'])):
        # Plot matched markers
        idx = np.where(sg_catalogue == peak_detection['subs_within'][i])[0]

        for j in idx:
            #ax.annotate(str(subfind_coords['idx'][idx][0]), xy=(subfind_coords['x'][idx], subfind_coords['y'][idx]), xytext=(-5, 5), textcoords='offset points', size=7)
            if legend_trigger:
                circle_1 = mpatches.Circle((subfind_coords['x'][idx], subfind_coords['y'][idx]), radius=0.2, color='k', alpha=(0.5 + alphas[i]/2), label = legends[2])
            else:
                circle_1 = mpatches.Circle((subfind_coords['x'][idx], subfind_coords['y'][idx]), radius=0.2, color='k', alpha=(0.5 + alphas[i]/2))
            ax.add_patch(circle_1)
        legend_trigger = False
    # Plot legend.
    # lgnd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #                  borderaxespad=0., numpoints=1, fontsize=16, shadow=True, fancybox=True,
    #                  facecolor = 'white', edgecolor = 'g')

    # #change the marker size manually for both lines
    # lgnd.legendHandles[0]._sizes = [30]
    # lgnd.legendHandles[1]._sizes = [30]
    ax.set_xlabel(ax_label[0])
    ax.set_ylabel(ax_label[1])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('Subhalo_match.pdf')
    #plt.show()
