"""
alpha = arctan(ky/kz)

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

basedir = Path("data/")
runs = [21, 15, 11, 12, 16, 13, 17, 14, 18]
R = [np.sqrt(0.5), 0.8, 1.01, 1.2, np.sqrt(1.75), np.sqrt(2.0), 1.5, np.sqrt(2.5), 2.0]
ssc = np.array(R)**2
alpha = []

growth = []
for run in runs:
    dfname = Path("run_{:d}_output.h5".format(run))
    df = h5py.File(basedir/dfname, "r")
    #alpha.append(np.arctan2(df['gamma'].attrs['max ky'], df['gamma'].attrs['max kz']))
    kk = np.sqrt(df['gamma'].attrs['max ky']**2 + df['gamma'].attrs['max kz']**2)
    alpha.append(np.arcsin(df['gamma'].attrs['max ky']/kk))
    growth.append(df['gamma'].attrs['max growth rate'])
    df.close()
plt.scatter(ssc, alpha, marker='o', c=growth)
plt.xlabel(r"$S/S_c$")
plt.ylabel(r"$\alpha$")
plt.savefig("plots/alpha_vs_ssc.png")
