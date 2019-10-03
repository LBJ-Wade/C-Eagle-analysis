import numpy as np
from matplotlib import pyplot as plt
import sys, platform, os
import matplotlib
import camb
from camb import model, initialpower

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
l_max = 10**4
pars.set_for_lmax(l_max, lens_potential_accuracy=0);
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)
totCL=      powers['total']
unlensedCL= powers['unlensed_scalar']
ls = np.arange(totCL.shape[0])

def fftIndgen(n):
    a = list(range(0, n//2+1))
    b = list(range(1, n//2))
    b = list(reversed(b))
    b = [-i for i in b]
    return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)


out = gaussian_random_field(Pk = np.array(i for i in totCL[:,0]), size=256)
plt.figure()
plt.imshow(out.real, interpolation='none')
plt.show()