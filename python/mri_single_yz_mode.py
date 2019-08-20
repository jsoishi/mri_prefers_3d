"""
The magnetorotational instability prefers three dimensions.

Single mode calculation. Returns spectrum.

Usage:
    mri_single_yz_mode.py [--ideal] <config_file> <ky> <kz>

Options:
    --ideal    Use Ideal MHD
"""

from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import os
import h5py
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

args = docopt(__doc__)
ky = float(args['<ky>'])
kz = float(args['<kz>'])
ideal = args['--ideal']
filename = Path(args['<config_file>'])
outbase = Path("data")

logger.info("Solving for mode ky = {}, kz = {}".format(ky, kz))
# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))

logger.info('Running mri.py with the following parameters:')
logger.info(config.items('parameters'))
try:
    dense = config.getboolean('solver','dense')
except:
    dense = False
if dense:
    logger.info("Using dense solver.")
    dense_threshold = config.getfloat('solver','dense_threshold')
else:
    logger.info("Using sparse solver.")

Nx = config.getint('parameters','Nx')
Lx = eval(config.get('parameters','Lx'))
B = config.getfloat('parameters','B')

Nmodes = config.getint('parameters','Nmodes')

R      =  config.getfloat('parameters','R')
q      =  config.getfloat('parameters','q')

kymin = config.getfloat('parameters','kymin')
kymax = config.getfloat('parameters','kymax')
Nky = config.getint('parameters','Nky')

kzmin = config.getfloat('parameters','kzmin')
kzmax = config.getfloat('parameters','kzmax')
Nkz = config.getint('parameters','Nkz')

ν = config.getfloat('parameters','ν')
η = config.getfloat('parameters','η')

kx     =  np.pi/Lx
S      = -R*B*kx*np.sqrt(q)
f      =  R*B*kx/np.sqrt(q)
cutoff =  kx*np.sqrt(R**2 - 1)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
domain = de.Domain([x_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

# 3D MRI

problem_variables = ['p','vx','vy','vz','bx','by','bz']
if not ideal:
    problem_variables += ['ωy','ωz','jxx']
problem = de.EVP(domain, variables=problem_variables, eigenvalue='gamma')
problem.meta[:]['x']['dirichlet'] = True

# Local parameters

problem.parameters['S'] = S
problem.parameters['f'] = f
problem.parameters['B'] = B

problem.parameters['ky'] = ky
problem.parameters['kz'] = kz

if not ideal:
    problem.parameters['ν'] = ν
    problem.parameters['η'] = η

# Operator substitutions for y,z, and t derivatives

problem.substitutions['dy(A)'] = "1j*ky*A"
problem.substitutions['dz(A)'] = "1j*kz*A"
problem.substitutions['Dt(A)'] = "gamma*f*A + S*x*dy(A)"

# Variable substitutions

if not ideal:
    problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
    problem.substitutions['jx'] = "dy(bz) - dz(by)"
    problem.substitutions['jy'] = "dz(bx) - dx(bz)"
    problem.substitutions['jz'] = "dx(by) - dy(bx)"
    probelm.substitutions['L(A)'] = "d(A,y=2) + d(A,z=2)"

# Hydro equations: p, vx, vy, vz, ωy, ωz

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")
if ideal:
    problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) = 0")
    problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) = 0")
    problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) = 0")

    # Frozen-in field
    problem.add_equation("Dt(bx) - B*dz(vx)        = 0")
    problem.add_equation("Dt(by) - B*dz(vy) - S*bx = 0")
    problem.add_equation("Dt(bz) - B*dz(vz)        = 0")

else:
    problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = 0")
    problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = 0")
    problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = 0")

    problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
    problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

    # MHD equations: bx, by, bz, jxx
    problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0")
    problem.add_equation("Dt(bx) - B*dz(vx) + η*( dy(jz) - dz(jy) )            = 0")
    problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - η*( dx(jxx) + L(jx) ) = 0")
    problem.add_equation("jxx - dx(jx) = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_bc("left(vx)   = 0")
problem.add_bc("right(vx)  = 0")
if not ideal:
    problem.add_bc("left(bx)   = 0")
    problem.add_bc("left(ωy)   = 0")
    problem.add_bc("left(ωz)   = 0")
    problem.add_bc("left(jxx)  = 0")

    problem.add_bc("right(bx)  = 0")
    problem.add_bc("right(ωy)  = 0")
    problem.add_bc("right(ωz)  = 0")
    problem.add_bc("right(jxx) = 0")

# GO

solver = problem.build_solver()
def ideal_2D(kz):
    kk, BB = kz*kz, B*B
    a = kx*kx + kk
    b = kk*(2*BB*a + f*(f+S))
    c = (BB*kk**2)*(BB*a + f*S)
    return np.sqrt( ( - b + np.sqrt(b*b - 4*a*c + 0j) ) / (2*a) )

t1 = time.time()
solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
t2 = time.time()

logger.info("Solve time: {}".format(t2-t1))

gamma = solver.eigenvalues
dense_threshold=1
gamma = gamma[np.abs(gamma) < dense_threshold]
index = np.argsort(-gamma.real)
gamma = gamma[index]
print("saving eigenvector with gamma = {}".format(gamma[0]))
nvars = len(problem_variables)
eigvec = np.zeros((nvars,Nx),dtype=np.complex128)

for k in range(nvars):
    solver.set_state(index[0])
    eigvec[k,:] = solver.state[problem_variables[k]]['g']

# Save either or both eigenvalues and eigenvectors to a single .h5 file
# Output file will be the .cfg file name with _output.h5
if CW.rank == 0:
    tail = '_single_mode.h5'
    if ideal:
        tail = '_ideal' + tail
    output_file_name = Path(filename.stem + tail)
    output_file = h5py.File(outbase/output_file_name, 'w')
    dset_eval = output_file.create_dataset('eigvals',data=solver.eigenvalues)
    dset_eval.attrs.create("ky", ky)
    dset_eval.attrs.create("kz", kz)
    dset_eval.attrs.create("R", R)
    dset_eval.attrs.create("B", B)
    dset_eval.attrs.create("q", q)
    dset_eval.attrs.create("d", Lx)
    dset_evec = output_file.create_dataset('eigvecs',data=eigvec)
    dset_evec.attrs.create('x',x_basis.grid())
    
