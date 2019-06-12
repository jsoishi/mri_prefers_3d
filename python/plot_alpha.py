"""
alpha = arctan(ky/kz)

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

basedir = Path("data/")
runs = [1, 2, 3]
ssc = np.array([1.01, 1.2, 2.0])**2
alpha = []

for run in runs:
    dfname = Path("run_{:d}_output.h5".format(run))
    df = h5py.File(basedir/dfname, "r")
    alpha.append(np.arctan2(df['gamma'].attrs['max ky'], df['gamma'].attrs['max kz']))
    df.close()

plt.plot(ssc, alpha,'ko')
plt.xlabel(r"$S/S_c$")
plt.ylabel(r"$\alpha$")
plt.savefig("plots/alpha_vs_ssc.png")
