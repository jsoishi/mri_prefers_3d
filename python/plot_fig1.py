import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from calc_asymptotic_approx import calc_asymptotic_growth

plt.style.use('prl')

# params
filebase = 'data/run_{:d}_output.h5'
runs = [15, 39, 12, 13]
SSC = [0.64, 1.002001, 1.0201, 2.]
q = 0.75
index = 0

fig = plt.figure(figsize=(16,8))
# grid = AxesGrid(fig, (0.125, 0.5,0.8,0.35),
#                 nrows_ncols=(1, 4),
#                 axes_pad=0.1,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.1
#                 )
xgutter = 0.1
pad = 0.01
w = (1. - xgutter*2 - pad*3)/4
print(w)
h = 0.4
grid = [fig.add_axes([xgutter+i*(w+pad),0.55,w,h]) for i in range(4)]
for i,r in enumerate(runs):
    ax = grid[i]
    print("loading data from {}".format(filebase.format(r)))
    datafile = h5py.File(filebase.format(r),'r')
    ky_global    = datafile['ky']
    kz_global    = datafile['kz']
    gamma_global = datafile['gamma'][:]
    gamma_r = gamma_global.real/q # scale by S
    #gamma_r[np.where(gamma_r<0)] = 0.0
    max_gamma = datafile['gamma'].attrs['max growth rate']
    max_ky = datafile['gamma'].attrs['max ky']
    max_kz = datafile['gamma'].attrs['max kz']
    contour_levels = np.linspace(0,gamma_r.max(),5)
    vmin = 0.
    vmax = 0.1
    c = ax.imshow(gamma_r, extent=[kz_global[:].min(),kz_global[:].max(),ky_global[:].min(),ky_global[:].max()], vmin=vmin,vmax=vmax, origin='lower')

    ax.contour(kz_global,ky_global,gamma_r,levels=contour_levels,colors='k')
    ax.contour(kz_global,ky_global,gamma_r,levels=[0.],colors='grey')
    ax.text(0.05,0.9,r'$S/S_c = {:5.3f}$'.format(SSC[i]), color='w',fontsize=18)
    ax.plot(max_kz, max_ky, 'ro')
    #ax.colorbar(PCM, label=r'$\gamma$')
    ax.set_xlabel(r'$k_z$')

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    if i > 0:
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel(r'$k_y$')
        
    datafile.close()

cax = fig.add_axes([xgutter+4*(w+pad), 0.55,0.02,h])
cbar = fig.colorbar(c, cax=cax)
cbar.set_label(r'$\gamma/S$')

# asymptotics
xloc = grid[0].get_position().x0
w_lower = 2*w + pad
asymp_ax = fig.add_axes([0.1,0.1,w_lower,0.35])
asymp_sub = asymp_ax.inset_axes([0.6,0.25,0.3,0.3*w_lower/0.35])

filename = 'data/run_41_output.h5'
data = h5py.File(filename, "r")
gamma_global = data['gamma'][:]
gamma_r = gamma_global.real
gamma_i = gamma_global.imag
ky_global    = data['ky']
kz_global    = data['kz']

ky = ky_global[:]
kz = kz_global[index]
grid[1].axvline(kz, ymin=0, ymax=0.2, alpha=0.7)
R = 1.001
B = 1
d = np.pi
q = 0.75
ky_an = np.linspace(0.0,0.2,400)
omega, omega2 = calc_asymptotic_growth(R, q, B, d, ky_an, kz)

asymp_ax.plot(ky,gamma_r[:,index]/q, label='numerical')
asymp_ax.plot(ky_an, omega/q, label='asymptotic')
asymp_sub.plot(ky,-(gamma_r[:,index]/q + 1j*gamma_i[:,index]/q)**2, label='numerical')
asymp_sub.plot(ky_an, omega2/q**2, label='asymptotic')

#asymp_ax.legend(loc='upper right')
asymp_ax.text(0.125, 0.048, 'numerical', fontsize=18)
asymp_ax.text(0.035, 0.03, 'asymptotic', fontsize=18)
asymp_ax.set_ylim(0, 0.06/q)
asymp_ax.set_xlim(0,0.2)
asymp_ax.set_xlabel(r'$k_y$')
asymp_ax.set_ylabel(r'$\gamma/S$')
asymp_ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
asymp_sub.set_ylim(-0.005/q**2, 0.005/q**2)
asymp_sub.set_xlim(0.03,0.09)
asymp_sub.set_xlabel(r'$k_y$')
asymp_sub.set_ylabel(r'$\sigma^2/S^2$')
# spectrum

grid[1].annotate("",
            xy=(0.45, 0.26), xycoords='data',
            xytext=(0.55, 0.45), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3",
                            color='k',
                            alpha=0.4),
            )

spec_ax = fig.add_axes([xgutter+w_lower+pad,0.1,w_lower,0.35])

filename = 'data/run_39_single_mode.h5'
data = h5py.File(filename, 'r')
spectrum = data['eigvals'][:]

thresh = 2
spec_thresh = spectrum[np.abs(spectrum) < thresh]/q
spec_ax.scatter(spec_thresh[(spec_thresh.real < 0) & (np.abs(spec_thresh.imag) < 1e-8)].real,spec_thresh[(spec_thresh.real < 0) & (np.abs(spec_thresh.imag) < 1e-8)].imag,zorder=10)
spec_ax.scatter(spec_thresh[spec_thresh.real > 0].real, spec_thresh[spec_thresh.real > 0].imag)
spec_ax.scatter(spec_thresh[spec_thresh.real <= 0].real, spec_thresh[spec_thresh.real <= 0].imag, color='lightgrey',zorder=3)

spec_ax.set_xlim(-0.25/q,0.1/q)
spec_ax.set_ylim(-0.8/q,0.8/q)
spec_ax.set_xlabel(r"$\gamma/S$")
spec_ax.set_ylabel(r"$\omega/S$")
spec_ax.yaxis.tick_right()
spec_ax.yaxis.set_label_position("right")
spec_ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))

fig.savefig('plots/fig_1.pdf')#,bbox_inches='tight')



