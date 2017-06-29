""" 
Test scattering and gathering a multi-dimensional array in MPI.
"""

import sys
import numpy as np
import copy
import lyafxcorr_kg as xcorr
import time
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    # Read in sample of realizations, and rearrange indices so that the
    #iterations are in the first row
    fullarr_in   = np.load("covar_mosdef_n400_1.npy")
    fullarr      = np.rollaxis(fullarr_in, 2,0)
    # Flatten
    fullarr_flat = np.ndarray.flatten(fullarr)
    nvec_percore    = len(fullarr_flat)/size
    gathered_flat = np.empty(len(fullarr_flat))
else:
    fullarr      = None
    fullarr_in   = None
    fullarr_flat = None
    nvec_percore = None

comm.Bcast(nvec_percore, root=0) 

arr_local = np.zeros(nvec_percore)

comm.Scatter(fullarr_flat, arr_local, root=0)

comm.Barrier()

gathered_flat = comm.gather(arr_local, root=0)

if rank == 0:
    gathered_reshape = np.reshape(gathered_flat, np.shape(fullarr))
    full_gathered = np.rollaxis(gathered_reshape, 1,0)
    full_gathered = np.rollaxis(full_gathered, 2,1)
    np.array_equal(

exit()
