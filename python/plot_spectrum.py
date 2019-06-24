import numpy as np
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

outbase = Path("plots")
filename = Path(sys.argv[-1])
data = h5py.File(filename, 'r')

thresh = 2
spectrum = data['eigvec'][:]

spec_thresh = spectrum[np.abs(spectrum) < thresh]
plt.scatter(spec_thresh[spec_thresh.real <= 0].real, spec_thresh[spec_thresh.real <= 0].imag)
plt.scatter(spec_thresh[spec_thresh.real > 0].real, spec_thresh[spec_thresh.real > 0].imag, color='red')
plt.xlim(-0.25,0.25)
plt.ylim(-0.8,0.8)
plt.xlabel("real")
plt.ylabel("imag")

plot_file_name = Path(filename.stem + '_spectrum.png')
plt.savefig(outbase/plot_file_name, dpi=300)
