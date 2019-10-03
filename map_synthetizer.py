from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))



def map_mass(x,y,z, show_trigger = False, save_trigger = False):
	"""
	INPUTS:
	show_trigger: default = False
		if set to true triggers the plt.show() and prints the map onto the screen.
	save_trigger: default = False
		if set to True triggers the plt.savefig(*args) and saves the figure in the 
		current working directory.

	RETURNS: the x,y,z, values used for generating the plot in plt.show().
		e.g. x_data ,y_data ,z_data = map_mass(*args)
	"""

	from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


	fig, axes = plt.subplots(1, 1, figsize=(17, 8), sharey=False, sharex=False, squeeze=False)
	#plt.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.1)


	nbins = 800
	font_size = 24
	# line of sight momentum weights
	H, xbins, ybins = np.histogram2d(x, y, bins=(nbins, nbins), weights = z)
	lt = get_centers_from_bins(xbins)
	lm = get_centers_from_bins(ybins)
	mesh_X, mesh_Y = np.meshgrid(lt, lm)
	cmap = 'viridis'
	#norm = MidpointNormalize(vmin=H.T.min(), vmax=H.T.max(), midpoint=0)
	#img = axes.pcolor(mesh_X, mesh_Y, H.T, cmap=cma,p norm= norm,  alpha=1)	
	img = axes.pcolor(mesh_X, mesh_Y, H.T, cmap=cmap, alpha=1)
	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes)
	cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
	cbar.set_label('n_x[ii]', labelpad = -74, fontsize=font_size)
	#	cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")
	cax2.xaxis.set_tick_params(labelsize=font_size-6)


	if show_trigger: plt.show()
	if save_trigger: plt.savefig('File.pdf')


def get_centers_from_bins(bins):
    """ return centers from bin sequence """
    return (bins[:-1] + bins[1:])/2
    #return bins[:]

def get_centers_from_log_bins(bins):
    """ return centers from bin sequence """
    return np.sqrt(bins[:-1]*bins[1:])

def bins_meshify(x, y, x_bins, y_bins):
	"""

	"""
	_ , xbins, ybins = np.histogram2d(x, y, bins=(x_bins, y_bins), weights = None)
	lt = get_centers_from_bins(xbins)
	lm = get_centers_from_bins(ybins)
	cX_v, cY_v = np.meshgrid(lt, lm)
	return cX_v, cY_v

def bins_evaluate(x, y, x_bins, y_bins, weights = None):
	"""

	"""
	H , _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins), weights = weights)
	return H.T



def plot_density_map(x, y, xbins, ybins, Nlevels=4, cbar=True, weights=None):

    Z = np.histogram2d(x, y, bins=(xbins, ybins), weights=weights)[0].astype(float).T

    # central values
    lt = get_centers_from_bins(xbins)
    lm = get_centers_from_bins(ybins)
    cX, cY = np.meshgrid(lt, lm)
    X, Y = np.meshgrid(xbins, ybins)

    im = plt.pcolor(X, Y, Z, cmap=plt.cm.Blues)
    plt.contour(cX, cY, Z, levels=nice_levels(Z, Nlevels), cmap=plt.cm.Greys_r)

    if cbar:
        cb = plt.colorbar(im)
    else:
        cb = None
    plt.xlim(xbins[0], xbins[-1])
    plt.ylim(ybins[0], ybins[-1])

    try:
        plt.tight_layout()
    except Exception as e:
        print(e)
    return plt.gca(), cb 


def map_interpolate(Data, Pow2factor, kind, disp=False):
	"""

	"""
	from scipy import interpolate
	if disp:
	    p30 = np.poly1d(np.polyfit([10, 30, 50, 60, 70, 80], np.log([0.8, 1.3, 6, 12, 28, 56]), 2))
	    print('Expectd time for NxN data:', np.exp(p30(Data.shape[0])))
	x = np.arange(Data.shape[1])
	y = np.arange(Data.shape[0])
	xv, yv = np.meshgrid(x, y)
	f = interpolate.interp2d(xv, yv, Data, kind=kind)

	xnew = np.arange(0, Data.shape[1], 1 / (2**Pow2factor))
	ynew = np.arange(0, Data.shape[0], 1 / (2**Pow2factor))
	Upsampled = f(xnew, ynew)

	return Upsampled 


def plot_circle(axes, x,y,r):
	from matplotlib.patches import Circle
	axes.add_artist(Circle((x,y), r))


def modified_spectral_cmap(Reversed = False):
	import matplotlib.colors as mcolors
	# sample the colormaps that you want to use. Use 128 from each so we get 256
	# colors in total
	if not Reversed:
		colors1 = plt.cm.Spectral(np.linspace(0., 0.5, 127))
		colors2 = plt.cm.binary(np.linspace(0., 0.0001, 2))
		colors3 = plt.cm.Spectral(np.linspace(0.5, 1, 127))
	else: 
		colors1 = plt.cm.Spectral_r(np.linspace(0., 0.5, 127))
		colors2 = plt.cm.binary(np.linspace(0., 0.0001, 2))
		colors3 = plt.cm.Spectral_r(np.linspace(0.5, 1, 127))
	# combine them and build a new colormap
	colors = np.vstack((colors1, colors2, colors3))
	mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
	return mymap

def Martin_Dream_CMap(N = 256, Reversed = False):
	dream_colors = np.ones((int(N),4))
	dream_colors[:int(N/2),0] = np.linspace(160/255,52/255,int(N/2))
	dream_colors[:int(N/2),1] = np.linspace(160/255,82/255,int(N/2))
	dream_colors[:int(N/2),2] = np.linspace(160/255,255/255,int(N/2))
	dream_colors[int(N/2):,0] = np.linspace(52/255,77/255,int(N/2))
	dream_colors[int(N/2):,1] = np.linspace(82/255,6/255,int(N/2))
	dream_colors[int(N/2):,2] = np.linspace(255/255,122/255,int(N/2))

	if Reversed:
		dream_colors = np.flip(dream_colors, 0)

	from matplotlib.colors import ListedColormap

	dream_cmap = ListedColormap(dream_colors)

	return dream_cmap


def Manchester_CMap(N = 256, Reversed = False):
	manchester_colors = np.ones((int(N),4))
	manchester_colors[:int(N/2),0] = np.linspace(160/255,255/255,int(N/2))
	manchester_colors[:int(N/2),1] = np.linspace(160/255,168/255,int(N/2))
	manchester_colors[:int(N/2),2] = np.linspace(160/255,0/255,int(N/2))
	manchester_colors[int(N/2):,0] = np.linspace(255/255,77/255,int(N/2))
	manchester_colors[int(N/2):,1] = np.linspace(168/255,6/255,int(N/2))
	manchester_colors[int(N/2):,2] = np.linspace(0/255,122/255,int(N/2))

	if Reversed:
		manchester_colors = np.flip(manchester_colors, 0)

	from matplotlib.colors import ListedColormap

	manchester_cmap = ListedColormap(manchester_colors)

	return manchester_cmap

def GreyMap(N = 256, Reversed = False):
	grey_colors = np.ones((int(N),4))
	grey_colors[:N,0] = np.linspace(200/255,0/255,N)
	grey_colors[:N,1] = np.linspace(200/255,0/255,N)
	grey_colors[:N,2] = np.linspace(200/255,0/255,N)

	if Reversed:
		grey_colors = np.flip(grey_colors, 0)

	from matplotlib.colors import ListedColormap

	grey_cmap = ListedColormap(grey_colors)

	return grey_cmap