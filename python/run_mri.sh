#!/usr/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=faculty

source /home/joishi/build/dedalus_intel_mpi/bin/activate
#conda activate dedalus

date
mpirun -np 100 python3 mri.py runs/run_1.cfg
date
