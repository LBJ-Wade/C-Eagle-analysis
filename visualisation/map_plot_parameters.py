def test_plot():
	import numpy as np
	from matplotlib import pyplot as plt
	mean = [0, 0]
	cov = [[1, 1], [1, 2]]
	x, y = np.random.multivariate_normal(mean, cov, 10000).T
	plt.hist2d(x, y, bins=30, cmap='Blues')
	cb = plt.colorbar()
	cb.set_label(r'$\mathrm{counts\ in\ bin}$')
	plt.show()

def test_plot_smoothed():
	from scipy.stats import gaussian_kde
	import numpy as np
	from matplotlib import pyplot as plt
	mean = [0, 0]
	cov = [[1, 1], [1, 2]]
	x, y = np.random.multivariate_normal(mean, cov, 10000).T
	# fit an array of size [Ndim, Nsamples]
	data = np.vstack([x, y])
	kde = gaussian_kde(data)

	# evaluate on a regular grid
	xgrid = np.linspace(-3.5, 3.5, 100)
	ygrid = np.linspace(-6, 6, 100)
	Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
	Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

	# Plot the result as an image
	plt.imshow(Z.reshape(Xgrid.shape),
	           origin='lower', aspect='auto',
	           extent=[-3.5, 3.5, -6, 6],
	           cmap='Blues')
	cb = plt.colorbar()
	cb.set_label("density")
	plt.show()



# Example of implementation
# set_defaults_plot()
# test_plot()
