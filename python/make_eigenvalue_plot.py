"""
This script plots both the largest growth rates of the MRI data
passed to the script. To run, pass it a file as follows:

python3 make_eigenvector_plots.py mri_dataset_845.h5

"""

import h5py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('prl')

# Parses filename passed to script
parser = argparse.ArgumentParser(description='Passes filename')
parser.add_argument('filename', metavar='Rc', type=str, help='.h5 file to plot eigenvectors for maximum eigenvalue')
args = parser.parse_args()
filename = Path(vars(args)['filename'])
outbase = Path("plots")

# Plot growth rates
datafile = h5py.File(filename,'r')
gamma_global = datafile['gamma'][:]
gamma_r = gamma_global.real
#gamma_r[np.where(gamma_r<0)] = 0.0
#max_gamma = datafile['gamma'].attrs['max growth rate']
ky_global    = datafile['ky']
kz_global    = datafile['kz']
print(ky_global.shape)
print(gamma_r.shape)
try:
    max_ky = datafile['gamma'].attrs['max ky']
    max_kz = datafile['gamma'].attrs['max kz']
except KeyError:
    ky_index, kz_index = np.where(gamma_r == gamma_r.max())
    max_ky = ky_global[ky_index[0]]
    max_kz = kz_global[kz_index[0]]

print("max growth rate on grid = {}".format(gamma_r.max()))

contour_levels = np.linspace(0,gamma_r.max(),10)
vmin = 0.
vmax = 0.071
PCM = plt.pcolormesh(kz_global,ky_global,gamma_r, vmin=vmin,vmax=vmax)

plt.contour(kz_global,ky_global,gamma_r,levels=contour_levels,colors='k')
plt.contour(kz_global,ky_global,gamma_r,levels=[0.],colors='w')
plt.plot(max_kz, max_ky, 'ro')
plt.colorbar(PCM, label=r'$\gamma$')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$k_y$')
plt.tight_layout()

#plt.title(r'3D Keplerian MRI growth rates/f  $\left( S/S_{\mathrm{crit.}} = %.2f\right)$' %(R[i]))
plot_file_name = Path(filename.stem + '_growthrates.png')
plt.savefig(outbase/plot_file_name, dpi=300)
