from astropy.cosmology import FlatLambdaCDM
import numpy as np
from matplotlib import pyplot as plt
from numpy import logspace


def cosmo_params():
	h = 0.6777
	Omega_matter = 0.16
	return h*100, Omega_matter


def luminosity_D(redshift):
	""" Luminosity distance in Mpc at redshift `z`.

        This is the distance to use when converting between the
        bolometric flux from an object at redshift `z` and its
        bolometric luminosity.

        Parameters
        ----------
        z : array_like
          Input redshifts.

        Returns
        -------
        d : ndarray, or float if input scalar
          Luminosity distance in Mpc at each input redshift.

        References
        ----------
        Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
	"""
	H0, Omega_matter = cosmo_params()
	cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_matter)
	D = cosmo.luminosity_distance(redshift)
	return D.value

def angular_diameter_D(redshift):
	""" Angular diameter distance in Mpc at a given redshift.

        This gives the proper (sometimes called 'physical') transverse
        distance corresponding to an angle of 1 radian for an object
        at redshift `z`.

        Weinberg, 1972, pp 421-424; Weedman, 1986, pp 65-67; Peebles,
        1993, pp 325-327.

        Parameters
        ----------
        z : array_like
          Input redshifts.

        Returns
        -------
        d : ndarray, or float if input scalar
          Angular diameter distance in Mpc at each input redshift.
	"""
	H0, Omega_matter = cosmo_params()
	cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_matter)
	D = cosmo.angular_diameter_distance(redshift)
	return D.value

def test():
	"""
	EXPECTED OUTPUT:

	luminosity_distance:  3417.3640878635133
	angular_diameter_distance:  1386.4108433865529
	"""
	print("luminosity_distance: ", luminosity_D(0.57))
	print("angular_diameter_distance: ", angular_diameter_D(1.75)*(8.6/3600*np.pi/180))


def plots():

	z = logspace(-5, 3.2, 100)
	D_l = luminosity_D(z)
	D_a = angular_diameter_D(z)

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
	axes[0].set_xscale('log')
	axes[0].set_yscale('log')
	axes[0].plot(z, D_a, label=r'$\mathrm{Angular\ diameter\ distance}$')
	axes[0].plot(z, D_l, label=r'$\mathrm{Luminosity\ distance}$')
	axes[0].set_xlabel(r'$z$')
	axes[0].set_ylabel(r'$d_L, d_A\qquad [Mpc]$')
	axes[0].legend()
	axes[0].grid()

	axes[1].set_xscale('log')
	axes[1].set_yscale('log')
	axes[1].plot(1+z, D_a/D_l)
	axes[1].set_xlabel(r'$z$')
	axes[1].set_ylabel(r'$d_A/d_L$')
	axes[1].grid()

	#plt.savefig('distance_cosmology.pdf')
	#plt.show()
# test()
#plots()