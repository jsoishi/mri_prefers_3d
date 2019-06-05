import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Parses filename passed to script
parser = argparse.ArgumentParser(description='Passes filename')
parser.add_argument('filename', metavar='Rc', type=str, help='.h5 file to plot eigenvectors for maximum eigenvalue')
args = parser.parse_args()
filename = vars(args)['filename']

x = np.linspace(1,128,128)

spn = 1
eigenvector_vars = ['p','vx','vy','vz','ωy','ωz','bx','by','bz','jxx']

# Import .h5 file passed to script 
datafile = h5py.File(filename,'r')
eigvec = datafile['eigvec'][:,:,:,:]
gamma_global = datafile['gamma'][:,:]
eigvec_real = eigvec.real
eigvec_imag = eigvec.imag

# Find grid point with largest growth rate to plot corresponding eigenvectors
gamma_r = gamma_global.real
gamma_r[np.where(gamma_r<0)] = 0.0
max_point = np.amax(gamma_r)
z_max = np.where(gamma_r == max_point)[1][0]
y_max = np.where(gamma_r == max_point)[0][0]
zv = kz_global[z_max] #max x value
yv = ky_global[y_max] #max y value
maxvalues[0,i] = zv
maxvalues[1,i] = yv

plt.figure(figsize=(10,16))

# Create subplots
for i in range(10):
	plt.subplot(5,2,spn)
	plt.title('$' + eigenvector_vars[i] + '$')
	plt.plot(x,eigvec_real[y_max,z_max,i,:]) # Real values
	plt.plot(x,eigvec_imag[y_max,z_max,i,:],linestyle='dashed') # Imaginary values
	spn += 1
plt.tight_layout()
plot_file_name = filename + '_eigenvectors.png'
plt.savefig(plot_file_name, dpi=300)