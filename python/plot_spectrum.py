"""plot_spectrum.py [--R=<R>] <filename>

Usage:
    plot_spectrum.py [--R=<R> --kz=<kz> --ky=<ky>] <filename>

"""

import numpy as np
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

from docopt import docopt

# parse arguments
args = docopt(__doc__)
filename = args['<filename>']
R = float(args['--R'])
ky = float(args['--ky'])
kz = float(args['--kz'])

# check asymptotic calculation
if R:
    Rprime = R**2 - 1
    q = 0.75
    B = 1.
    d = np.pi
    omega2 = q*B**2/(q+1)*(kz**2*(d**2/np.pi**2 * kz**2 - Rprime) - ((6 + np.pi**2)*q + np.pi**2 - 6)*ky**2/12.)
    omega = np.sqrt(-omega2)
    print("asymptotic omega = {}".format(omega))

outbase = Path("plots")
filename = Path(filename)
data = h5py.File(filename, 'r')

thresh = 2
spectrum = data['eigvec'][:]

spec_thresh = spectrum[np.abs(spectrum) < thresh]
plt.scatter(spec_thresh[spec_thresh.real <= 0].real, spec_thresh[spec_thresh.real <= 0].imag)
plt.scatter(spec_thresh[spec_thresh.real > 0].real, spec_thresh[spec_thresh.real > 0].imag)
plt.gca().set_prop_cycle(None)
plt.plot(-omega, 0., marker='^')
plt.plot(omega, 0., marker='^')
plt.xlim(-0.25,0.25)
plt.ylim(-0.8,0.8)
plt.xlabel("real")
plt.ylabel("imag")

plot_file_name = Path(filename.stem + '_spectrum.png')
plt.savefig(outbase/plot_file_name, dpi=300)
