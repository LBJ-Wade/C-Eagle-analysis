import matplotlib.pyplot as plt
import astropy
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel
import numpy as np
from astropy.convolution.kernels import CustomKernel, Model2DKernel

def Airy_kernel(pixel_smoothing = 10):
	"""
	pixel_smoothing: integer value
		when performing the discrete convolution between a map and a kernel,
		the pixel smoothing is the number of pixel corresponding to the FWHM of the 
		kernel, mapped onto the bins of the input map.

	RETURN: an AstroPy 2D Kernel (Airy Disk mornalised to central value=1)

	EXTRA FUNCTIONALITIES: 
		access object type as type(kernel)
		convert AstroPy kernel to np 2D array as kernel.array
	"""
	airydisk_2D_kernel = AiryDisk2DKernel(pixel_smoothing)
	return airydisk_2D_kernel

def Gaussian_kernel(pixel_smoothing = 10):
	"""
	pixel_smoothing: integer value
		when performing the discrete convolution between a map and a kernel,
		the pixel smoothing is the number of pixel corresponding to the FWHM of the 
		kernel, mapped onto the bins of the input map.

	RETURN: an AstroPy 2D Kernel (Gaussian mornalised to central value=1)

	EXTRA FUNCTIONALITIES: 
		access object type as type(kernel)
		convert AstroPy kernel to np 2D array as kernel.array
	"""
	gaussian_2D_kernel = Gaussian2DKernel(pixel_smoothing)
	return gaussian_2D_kernel

def show_img_kernel(kernel):
	"""
	kernel: input an AstroPy 2D kernel
		This function displays the kernel as a density plot, using Matplotlib
		and imshow. The function also prints it out to the screen.

	RETURN: Nothing
	"""

	plt.imshow(kernel, interpolation='none', origin='lower')
	plt.xlabel('x [pixels]')
	plt.ylabel('y [pixels]')
	plt.colorbar()
	plt.show()

def show_3d_kernel(kernel):
	"""
	kernel: input an AstroPy 2D kernel
		This function displays the kernel as a 3d plot, using Matplotlib
		and surface plot. The function also prints it out to the screen.

	RETURN: Nothing
	"""
	kernel = kernel.array
	# Create an X-Y mesh of the same dimension as the 2D data. You can
	# think of this as the floor of the plot.
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib.colors import LightSource

	x_data, y_data = np.meshgrid( np.arange(kernel.shape[1]),
	                              np.arange(kernel.shape[0]) )

	x_data = x_data#.flatten()
	y_data = y_data#.flatten()
	z_data = kernel#.flatten()

	# Create a figure for plotting the data as a 3D histogram.
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Create light source object.
	ls = LightSource(azdeg=0, altdeg=90)
	# Shade data, creating an rgb array.
	rgb = ls.shade(np.log10(z_data+10e-7), plt.cm.coolwarm)
	surf = ax.plot_surface(x_data, y_data, z_data, rstride=1, cstride=1, linewidth=0,
	                       antialiased=True, facecolors=rgb, alpha=1)
	plt.grid(linestyle='--', alpha=0.2, color='k')
	plt.show()

def test_custom_kernel():
	array = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
	kernel = CustomKernel(array)
	print(kernel.dimension)
	custom_kernel = Model2DKernel(kernel, x_size=9)
	return custom_kernel

def nika2_kernel(xbins, ybins, kernel_Type = 'gauss'):
	# NIKA2 parameters [http://ipag.osug.fr/nika2/Instrument.html]
	# Frequency = 150 GHz

	fwhm = 17.7			# arcsec
	fov = 6.5			# arcmin, diameter $$ 
	
	xsize = xbins[1]-xbins[0]	# arcmin/pixel
	ysize = ybins[1]-ybins[0]	# arcmin/pixel

	xfwhm = fwhm/(60*xsize)
	yfwhm = fwhm/(60*ysize)

	xstdev = xfwhm/2.355
	ystdev = yfwhm/2.355

	#print("SMOOTHING CHECK --- x:", xstdev, "y:", ystdev)
	if xstdev == ystdev:
		if kernel_Type == 'gauss' or kernel_Type == 'Gauss':
			nika2kern = Gaussian2DKernel(xstdev)
		if kernel_Type == 'airy' or kernel_Type == 'Airy':
			nika2kern = AiryDisk2DKernel(xstdev)

	else:
		#print("WARNING: standard deviation different for each dimension")
		if kernel_Type == 'gauss' or kernel_Type == 'Gauss':
			nika2kern = Gaussian2DKernel(x_stddev = xstdev, y_stddev=ystdev)
		if kernel_Type == 'airy' or kernel_Type == 'Airy':
			nika2kern = AiryDisk2DKernel(xstdev)

	return nika2kern, fwhm

if __name__ == "__main__":
	# Example of implementation
	show_3d_kernel(Airy_kernel())