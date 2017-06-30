""" 
Test scattering and gathering a multi-dimensional array in MPI.
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    print("Reading in: bootsamp_mosdef_n400_1.npy")
    fullarr_in   = np.load("bootsamp_mosdef_n400_1.npy")
    print(np.shape(fullarr_in))
    fullarr      = np.rollaxis(fullarr_in, 2,0)
    # Flatten
    fullarr_flat = np.ndarray.flatten(fullarr)
    nvec_percore    = np.array(len(fullarr_flat)/size,'int')
    gathered_flat = np.empty(len(fullarr_flat))
    print("{:} elements per core".format(nvec_percore))
else:
    fullarr_in = None
    fullarr_in   = None
    fullarr_flat = None
    gathered_flat = None
    nvec_percore = None

nvec_percore = comm.bcast(nvec_percore, root=0) 

print("Rank {0:} is receiving {1:} elements".format(rank,nvec_percore))

arr_local = np.zeros(nvec_percore)

comm.Scatter(fullarr_flat, arr_local, root=0)

comm.Barrier()

comm.Gather(arr_local, gathered_flat, root=0)

if rank == 0:
    gathered_reshape = np.reshape(gathered_flat, np.shape(fullarr))
    full_gathered = np.rollaxis(gathered_reshape, 1,0)
    full_gathered = np.rollaxis(full_gathered, 2,1)
    np.array_equal(full_gathered, fullarr_in)
    np.save("bootsamp_test_out_mosdef_n400_1.npy", full_gathered)
