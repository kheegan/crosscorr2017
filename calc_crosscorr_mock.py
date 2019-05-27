""" 
Calculate Ly-alpha/galaxy cross-correlation for an ensemble of abs x galaxy realizations generated 
in May 2019. 

The input start and end indices refer to subvolumes of the TreePM box, ranging from 1 to 80 
inclusive. In each subvolume, we loop over 3x absorption realizations and 10x galaxy realizations.


"""

import sys
import numpy as np
import time as time
import lyafxcorr_kg as xcorr
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

istart = np.asarray(sys.argv[1]).astype(np.int32)
iend = np.asarray(sys.argv[2]).astype(np.int32)

# Define cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.31)

xcorr_dir = '/global/u1/k/kheegan/crosscorr2017/'

mockdir = xcorr_dir+'mocks/'

# Read in bin edges
PiBin_fil = xcorr_dir+'bins23_pi_0-30hMpc.txt'
SigBin_fil = xcorr_dir+'bins10_sigma_0-30hMpc.txt'

PiBins0 = ascii.read(PiBin_fil)
SigBins0 = ascii.read(SigBin_fil)

PiEdges = PiBins0['pi_edges'].data
SigEdges = SigBins0['sigma_edges'].data

# Convert bin boundaries from Mpc/h to Mpc
PiEdges  = PiEdges/(len(PiEdges)*[cosmo.h])
SigEdges = SigEdges/(len(SigEdges)*[cosmo.h])


for ivol in np.arange(istart,iend+1):

    for iabs in np.arange(0,3):

        for igal in np.arange(0,10):

            tstart = time.time()
            abssuffix = '{0:04d}'.format(ivol+iabs*80)

            lyafil = mockdir+'pixel_radecz_mock_'+abssuffix+'.bin'
            lyapix = xcorr.lyapix(lyafil,cosmo=cosmo)

            #print("Read in {0:d} Ly-a forest pixels from mock {1:03d}".format(lyapix.npix, 
            #                                                                  ivol))

            npix = lyapix.npix

            ### Read in galaxies 
            # We use the catalog created with Create_GalzMocks.IPYNB
            galsuffix = '{:03d}'.format(ivol+igal*80)
            galfil = mockdir+'cat_galmock_nonuniq_'+galsuffix+'.dat'

            gal = ascii.read(galfil, format='ipac')
            #print(gal.columns)

            gal_3d = gal[gal['source'] == '3DHST']
            gal_zD = gal[gal['source'] == 'zDeep']
            gal_mosdef = gal[gal['source'] == 'MOSDEF']
            gal_vuds = gal[gal['source']=='VUDS']
            gal_clamato=gal[gal['source']=='CLAMATO']

            #print('Read in %i 3D-HST galaxies, %i zCOSMOS-Deep galaxies, %i MOSDEF galaxies,' 
            #      '%i VUDS galaxies, %i CLAMATO galaxies' % 
            #      (len(gal_3d), len(gal_zD), len(gal_mosdef), len(gal_vuds),len(gal_clamato)) )

#            print("Calculating cross-correlation between galaxy catalog "+galsuffix+" and absorption catalog "+abssuffix)

            # Convert to 3D Sky positions
            Coord_3d     = SkyCoord(ra=gal_3d['ra']  , dec=gal_3d['dec']  ,
                                    distance=cosmo.comoving_distance(gal_3d['zspec']))
            Coord_zD     = SkyCoord(ra=gal_zD['ra']  , dec=gal_zD['dec']  ,
                                    distance=cosmo.comoving_distance(gal_zD['zspec']))
            Coord_mosdef = SkyCoord(ra=gal_mosdef['ra']  , dec=gal_mosdef['dec']  ,
                                    distance=cosmo.comoving_distance(gal_mosdef['zspec']))
            Coord_vuds   = SkyCoord(ra=gal_vuds['ra']  , dec=gal_vuds['dec']  ,
                                    distance=cosmo.comoving_distance(gal_vuds['zspec']))
            Coord_clamato=SkyCoord(ra=gal_clamato['ra'], dec=gal_clamato['dec']  ,
                                   distance=cosmo.comoving_distance(gal_clamato['zspec']))
            Coord_all = SkyCoord(ra=gal['ra'], dec=gal['dec']  ,
                                 distance=cosmo.comoving_distance(gal['zspec']))

            # Compute Cross-Correlation For MOSDEF Galaxies
            XCorr_mosdef, NoXCorr_mosdef = xcorr.xcorr_gal_lya(Coord_mosdef, 
                                                               lyapix, SigEdges, PiEdges, 
                                                               cosmo=cosmo, verbose=0)

            # Compute Cross-Correlation For zCOSMOS-Deep Galaxies
            XCorr_zD, NoXCorr_zD = xcorr.xcorr_gal_lya(Coord_zD, lyapix, 
                                                       SigEdges, PiEdges,cosmo=cosmo, verbose=0)


            # Compute Cross-Correlation for CLAMATO Galaxies
            XCorr_clamato, NoXCorr_clamato = xcorr.xcorr_gal_lya(Coord_clamato, lyapix, 
                                                                 SigEdges, PiEdges,cosmo=cosmo,
                                                                 verbose=0)

            # Compute Cross-Correlation For 3D-HST Galaxies
            XCorr_3d, NoXCorr_3d = xcorr.xcorr_gal_lya(Coord_3d, lyapix, 
                                                       SigEdges, PiEdges, cosmo=cosmo, verbose=0)


            # Compute Cross-Correlations for VUDS galaxies
            XCorr_vuds, NoXCorr_vuds = xcorr.xcorr_gal_lya(Coord_vuds, lyapix, 
                                                           SigEdges, PiEdges, cosmo=cosmo, 
                                                           verbose=0)
            
            filesuffix='g'+galsuffix+'_a'+abssuffix
            np.save(mockdir+"crosscorr/xcorrmock_3dhst_"+filesuffix+"_v4.1.npy", XCorr_3d)
            np.save(mockdir+"crosscorr/xcorrmock_zDeep_"+filesuffix+"_v4.1.npy", XCorr_zD)
            np.save(mockdir+"crosscorr/xcorrmock_mosdef_"+filesuffix+"_v4.1.npy", XCorr_mosdef)
            np.save(mockdir+"crosscorr/xcorrmock_vuds_"+filesuffix+"_v4.1.npy", XCorr_vuds)
            np.save(mockdir+"crosscorr/xcorrmock_clamato_"+filesuffix+"_v4.1.npy", XCorr_vuds)



 
