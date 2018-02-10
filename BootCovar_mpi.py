"""
Estimate covariance matrix of galaxy-Lya forest cross-correlations via 
bootstrap. This is a MPI version.

Takes an argument that points to an ASCII config file. 

Run as:
> python_mpi BootCovar_mpi.py input.cfg
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

if rank==0:
    t0 = time.clock()

    # Define cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.31)
    # Open config file and parse parameters
    input_fil = str(sys.argv[1]) 

    f=open(input_fil, 'r')
    lines = f.readlines()
    f.close()

    pixfil     = lines[0].partition('#')[0].rstrip()
    galfil     = lines[1].partition('#')[0].rstrip()
    cat_str    = lines[2].partition('#')[0].rstrip()
    PiBin_fil  = lines[3].partition('#')[0].rstrip()
    SigBin_fil = lines[4].partition('#')[0].rstrip()
    nsamp      = np.int(lines[5].partition('#')[0].rstrip())
    randseed_in= np.int(lines[6].partition('#')[0].rstrip())
    outsuffix  = lines[7].partition('#')[0].rstrip()

    # Read in forest pixels
    LyaPix = xcorr.lyapix(pixfil,cosmo=cosmo)
    print("Read in {0:d} Ly-a forest pixels".format(LyaPix.npix))
    npix = LyaPix.npix

    # Generate record of individual skewers
    SkewerRec = LyaPix.gen_SkewerRec()

    # Read in galaxy positions
    gal_table = ascii.read(galfil,format='ipac')
    gal = gal_table[gal_table['source'] == cat_str]
    print('Read in {0:d} galaxies'.format(len(gal)))

    # Generate 3D sky positions for galaxies
    GalCoords = SkyCoord(ra=gal['ra'],dec=gal['dec'],
                         distance=cosmo.comoving_distance(gal['zspec']))
    ngal = len(GalCoords)

    # Read in bin edges
    PiBins0 = ascii.read(PiBin_fil)
    SigBins0 = ascii.read(SigBin_fil)

    PiEdges = PiBins0['pi_edges'].data
    SigEdges = SigBins0['sigma_edges'].data

    # Convert bin boundaries from Mpc/h to Mpc
    PiEdges  = PiEdges/(len(PiEdges)*[cosmo.h])
    SigEdges = SigEdges/(len(SigEdges)*[cosmo.h])
    PiBound = (min(PiEdges), max(PiEdges))

    # Conformability test
    if (nsamp % size != 0):
        print("The number of cores must evenly divide nsamp")
        comm.Abort()

    # Number of bootstrap samples per node
    nsamp_node = np.int(nsamp/ size)

    # This is the array to receive gathered samples
    XCorrSamples_flat = np.empty(nsamp*(len(SigEdges)-1)*(len(PiEdges)-1))
else:
    LyaPix = None
    SkewerRec = None
    GalCoords = None
    ngal = None
    nsamp_node = None
    SigEdges = None
    PiEdges = None
    randseed_in= None
    cosmo = None
    XCorrSamples_flat = None

cosmo  = comm.bcast(cosmo, root=0)
LyaPix = comm.bcast(LyaPix,root=0)
SkewerRec = comm.bcast(SkewerRec, root=0)
GalCoords = comm.bcast(GalCoords,root=0)
ngal = comm.bcast(ngal,root=0)
nsamp_node = comm.bcast(nsamp_node, root=0)
SigEdges = comm.bcast(SigEdges, root=0)
PiEdges = comm.bcast(PiEdges, root=0)
randseed_in = comm.bcast(randseed_in, root=0)

# If RANDSEED_IN is positive, use it to initialize random number
# generator. Negative values will be ignored
#
# Results are reproducible only if input seed, rank AND 
# nsamp are the same!
if randseed_in > 0:
    np.random.seed(seed=randseed_in+rank)

# Initialize output array to store all the bootstrap samples
# This has the realizations stored in the leading dimension, which will
# need to be rolled before saving.
XCorrSamples_local = np.empty([nsamp_node, len(SigEdges)-1, len(PiEdges)-1])

# This is the loop over the desired resamples
for ii in range(0,nsamp_node):
    if (rank == 0) and (ii % 5 == 0):
        print("Iteration {} on rank 0".format(ii))
    # Make a copy of the pixels and resample
    LyaPixTmp = copy.deepcopy(LyaPix)
    LyaPixTmp = LyaPixTmp.resample_skewer(SkewerRec=SkewerRec)
    # Resample galaxy positions
    GalCoordTmp = GalCoords[np.random.choice(ngal,ngal,replace=True)]
    XCorrTmp, _ = xcorr.xcorr_gal_lya(GalCoordTmp, LyaPixTmp,SigEdges, PiEdges,
                                          cosmo=cosmo,verbose=0)
    XCorrSamples_local[ii,:,:] = XCorrTmp

XCorrSamples_local_flat = np.ndarray.flatten(XCorrSamples_local)
comm.Barrier()

comm.Gather(XCorrSamples_local_flat, XCorrSamples_flat, root=0)

if rank == 0:
    # Unflatten
    XCorrSamples = np.reshape(XCorrSamples_flat,
                              [nsamp, len(SigEdges)-1, len(PiEdges)-1])
    # Reshape to original shape for boostrap realizations
    XCorrSamples = np.rollaxis(XCorrSamples, 1, 0)
    XCorrSamples = np.rollaxis(XCorrSamples, 2, 1)
    # First, reshape the bootstrap samples so that it has dimensions (nbin, nsamp)
    XBootReshaped = XCorrSamples.reshape(-1, XCorrSamples.shape[-1])
    Covar = np.cov(XBootReshaped)

    outfil_cov = 'covar_'+outsuffix+'.npy'
    outfil_samp = 'bootsamp_'+outsuffix+'.npy'

    np.save(outfil_cov, Covar)
    np.save(outfil_samp, XCorrSamples)

    t1 = time.clock()
    print('Total runtime (s) = {}'.format(t1-t0))

exit()
