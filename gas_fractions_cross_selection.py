import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import gridspec

import map_plot_parameters as plotpar
#plotpar.set_defaults_plot()

# the random data
x = np.abs(np.random.randn(1000))
y = np.abs(np.random.randn(1000))

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

#plt.figure(1, figsize=(8, 8))
axScatter = plt.axes(rect_scatter)	# Define central plot
axHistx = plt.axes(rect_histx)		# Define horizontal marginal plot
axHisty = plt.axes(rect_histy)		# Define vertical marginal plot

# SELECTION
x_min, x_max = 0.5, +1
y_min, y_max = 0.5, +1
selection_color = 'yellow'
axScatter.axvspan(x_min, x_max, alpha=0.2, color=selection_color)
axScatter.axhspan(y_min, y_max, alpha=0.2, color=selection_color)
axHistx.axvspan(x_min, x_max, alpha=0.2, color=selection_color)
axHisty.axhspan(y_min, y_max, alpha=0.2, color=selection_color)

# LABELS & SCALES
axScatter.set_xlabel(r'$R/R_{200}$')
axScatter.set_ylabel(r'$f_g$')
axHistx.set_ylabel(r'$n_{SUB}$')
axHisty.set_xlabel(r'$n_{SUB}$')

axScatter.set_xscale('log'); axScatter.set_yscale('linear')
axHistx.set_xscale('log'); axHistx.set_yscale('log')
axHisty.set_xscale('log'); axHisty.set_yscale('linear')




# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y, s=1, c='k')

# now determine nice limits by hand:
nbins = 70
axScatter.set_xlim((0, 5))
axScatter.set_ylim((0, 5))
linbins = np.linspace(0, 5, nbins)
logbins = np.logspace(np.log10(10**-4), np.log10(5), nbins)
axHistx.hist(x, bins=logbins, color='k', histtype='step')
axHisty.hist(y, bins=linbins, orientation='horizontal', color='k', histtype='step')



axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()