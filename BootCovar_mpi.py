"""
Estimate covariance matrix of galaxy-Lya forest cross-correlations via 
bootstrap. 

Takes an argument that points to an ASCII config file. 

Run as:
> python BootCovar_script.py input.cfg
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

# If RANDSEED_IN is positive, use it to initialize random number
# generator. Negative values will be ignored

if randseed_in > 0:
    np.random.seed(seed=randseed_in)

# Read in forest pixels
LyaPix = xcorr.lyapix(pixfil,cosmo=cosmo)
print("Read in {0:d} Ly-a forest pixels".format(LyaPix.npix))
npix = LyaPix.npix

# Read in galaxy positions
gal_table = ascii.read(galfil)
gal = gal_table[gal_table['source'] == cat_str]
print('Read in {0:d} galaxies'.format(len(gal)))

# Generate 3D sky positions for galaxies
GalCoords = SkyCoord(ra=gal['ra']*u.degree,
                 dec=gal['dec']*u.degree,
                 distance=cosmo.comoving_distance(
                     gal['zspec']))

# Read in bin edges
PiBins0 = ascii.read(PiBin_fil)
SigBins0 = ascii.read(SigBin_fil)

PiEdges = PiBins0['pi_edges'].data
SigEdges = SigBins0['sigma_edges'].data

# Convert bin boundaries from Mpc/h to Mpc
PiEdges  = PiEdges/(len(PiEdges)*[cosmo.h])
SigEdges = SigEdges/(len(SigEdges)*[cosmo.h])
PiBound = (min(PiEdges), max(PiEdges))

# Initiate Bootstrap
ngal = len(GalCoords)

# Initialize output array to store all the bootstrap samples
XCorrSamples = np.empty([len(SigEdges)-1, len(PiEdges)-1, nsamp])

# This is the loop over the desired resamples
for ii in range(0,nsamp):
    if (ii % 5) == 0:
        print("Iteration #", ii)
    # Make a copy of the pixels and resample
    LyaPixTmp = copy.deepcopy(LyaPix)
    LyaPixTmp = LyaPixTmp.resample()
    # Resample galaxy positions
    GalCoordTmp = GalCoords[np.random.choice(ngal,ngal,replace=True)]
    XCorrTmp, _ = xcorr.xcorr_gal_lya(GalCoordTmp, LyaPixTmp,SigEdges, PiEdges,
                                      cosmo=cosmo,verbose=0)
    XCorrSamples[:,:,ii] = XCorrTmp

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
