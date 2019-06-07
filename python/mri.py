"""
The magnetorotational instability prefers three dimensions.
    
Dedalus script for calculating the maximum growth rates for the
3D MRI over a range of wavenumbers.

This script can be ran serially or in parallel, and produces an h5py file
that contains maximum eigenvalues and the corresponding eigenvectors.

Pass it a .cfg file to specify parameters for the particular solve.

To run using 4 processes, you would use:
    $ mpiexec -n 4 python3 mri.py mri_params.cfg

"""

import time
from configparser import ConfigParser
import argparse
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

# Parses .cfg filename passed to script
parser = argparse.ArgumentParser(description='Passes filename')
parser.add_argument('filename', metavar='Rc', type=str, help='.h5 file to plot eigenvectors for maximum eigenvalue')
args = parser.parse_args()
filename = vars(args)['filename']

# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(filename)

logger.info('Running mri.py with the following parameters:')
logger.info(config.items('parameters'))

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

problem_variables = ['p','vx','vy','vz','ωy','ωz','bx','by','bz','jxx']
problem = de.EVP(domain, variables=problem_variables, eigenvalue='gamma')
problem.meta[:]['x']['dirichlet'] = True

# Local parameters

problem.parameters['S'] = S
problem.parameters['f'] = f
problem.parameters['B'] = B

problem.parameters['ky'] = 0
problem.parameters['kz'] = 0.5*cutoff

problem.parameters['ν'] = ν
problem.parameters['η'] = η

# Operator substitutions for y,z, and t derivatives

problem.substitutions['dy(A)'] = "1j*ky*A"
problem.substitutions['dz(A)'] = "1j*kz*A"
problem.substitutions['Dt(A)'] = "gamma*f*A + S*x*dy(A)"

# Variable substitutions

problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['jx'] = "dy(bz) - dz(by)"
problem.substitutions['jy'] = "dz(bx) - dx(bz)"
problem.substitutions['jz'] = "dx(by) - dy(bx)"

# Hydro equations: p, vx, vy, vz, ωy, ωz

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = 0")
problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = 0")
problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = 0")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx

problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0")

problem.add_equation("Dt(bx) - B*dz(vx)            + η*( dy(jz) - dz(jy) )                   = 0")
problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - η*( dx(jxx) + dy(dy(jx)) + dz(dz(jx)) ) = 0")

problem.add_equation("jxx - dx(jx) = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_bc("left(vx)   = 0")
problem.add_bc("left(ωy)   = 0")
problem.add_bc("left(ωz)   = 0")
problem.add_bc("left(bx)   = 0")
problem.add_bc("left(jxx)  = 0")

problem.add_bc("right(vx)  = 0")
problem.add_bc("right(ωy)  = 0")
problem.add_bc("right(ωz)  = 0")
problem.add_bc("right(bx)  = 0")
problem.add_bc("right(jxx) = 0")

# GO

solver = problem.build_solver()

def ideal_2D(kz):
    kk, BB = kz*kz, B*B
    a = kx*kx + kk
    b = kk*(2*BB*a + f*(f+S))
    c = (BB*kk**2)*(BB*a + f*S)
    return np.sqrt( ( - b + np.sqrt(b*b - 4*a*c + 0j) ) / (2*a) )

# Create function to compute max growth rate for given ky, kz
def growth_rate(ky,kz,target,N=15):
    eigvec = np.zeros((10,Nx),dtype=np.complex128)
    # Change ky, kz parameters
    problem.namespace['ky'].value = ky
    problem.namespace['kz'].value = kz
    # Solve for eigenvalues with sparse search near target, rebuilding NCCs
    solver_failed = False
    try:
        solver.solve_sparse(solver.pencils[0], N=N, target=target, rebuild_coeffs=True)
    except:
        logger.info("Solver failed for (ky, kz) = (%f, %f)"%(ky, kz))
        solver_failed = True

    if solver_failed:
        gamma_r = np.nan
        gamma_i = np.nan
        gamma = []
        gamma.append(gamma_r + 1j*gamma_i)
    else:
        gamma = solver.eigenvalues
        index = np.argsort(-gamma.real)
        gamma = gamma[index]
    
        eigvec = np.zeros((10,Nx),dtype=np.complex128)

        for k in range(10):
            solver.set_state(index[0])
            eigvec[k,:] = solver.state[problem_variables[k]]['g']


        gamma_r = gamma.real[0]
        gamma_i = gamma.imag[0]
    
        if np.abs(gamma_r) <= 1e-6: gamma_r=0.0
        if np.abs(gamma_i) <= 1e-6: gamma_i=0.0
        
    logger.info('(ky,kz,gamma,omega) = (%f,%f,%f,%f)' %(ky,kz,gamma_r,gamma_i))
    
    # Return complex growth rate
    return gamma[0], eigvec

ky_global    = np.linspace(kymin,kymax,Nky)
kz_global    = np.linspace(kzmin,kzmax,Nkz)
gamma_global = np.zeros((Nky,Nkz),dtype=np.complex128)
eigvec_global = np.zeros((Nky,Nkz,10,Nx),dtype=np.complex128)

# Compute growth rate over local wavenumbers
kz_local    =    kz_global[CW.rank::CW.size]
gamma_local = gamma_global[:,CW.rank::CW.size]
eigvec_local = eigvec_global[:,CW.rank::CW.size]

t1 = time.time()
gamma_local[0] = ideal_2D(kz_local)
for i in range(1,Nky):
    for (k,kz) in enumerate(kz_local):
        soln = growth_rate(ky_global[i],kz,gamma_local[i-1,k], N=Nmodes)
        gamma_local[i,k] = soln[0]
        eigvec_local[i,k,:,:] = soln[1]
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Reduce growth rates to root process
gamma_global[:,CW.rank::CW.size] = gamma_local
eigvec_global[:,CW.rank::CW.size] = eigvec_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE,  gamma_global, op=MPI.SUM, root=0)
    CW.Reduce(MPI.IN_PLACE,  eigvec_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(gamma_global, gamma_global, op=MPI.SUM, root=0)
    CW.Reduce(eigvec_global, eigvec_global, op=MPI.SUM, root=0)

# Save either or both eigenvalues and eigenvectors to a single .h5 file
# Output file will be the .cfg file name with _output.h5
if CW.rank == 0:
    output_file_name = filename[0:-4] + '_output.h5'
    output_file = h5py.File(output_file_name, 'w')
    if config.getboolean('output','gamma') == True:
        dset = output_file.create_dataset('gamma',data=gamma_global)
    if config.getboolean('output','eigvec') == True:
        dset = output_file.create_dataset('eigvec',data=eigvec_global)
    output_file.close()
