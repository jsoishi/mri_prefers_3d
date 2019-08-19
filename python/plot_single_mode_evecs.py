import matplotlib.pyplot as plt
import h5py
import sys
from pathlib import Path

plt.style.use('prl')
filename = Path(sys.argv[-1])

rows = 2
cols = 3
names = [r'$v_x$',r'$v_y$',r'$v_z$',r'$b_x$',r'$b_y$',r'$b_z$']
plt.figure(figsize=(12,6))
with h5py.File(filename, 'r') as df:
    x = df['eigvecs'].attrs['x']
    for i in range(1,7):
        plt.subplot(rows,cols,i)
        plt.plot(x, df['eigvecs'][i,:].real, label='real')
        plt.plot(x, df['eigvecs'][i,:].imag, label='imag')
        if i == 1:
            plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(names[i-1])

plt.tight_layout()

plt.savefig('plots/' + filename.stem + 'eigvecs.pdf')

