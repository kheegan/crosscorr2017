"""Generate mock skewers from the TreePM256 simulation box for a relatively 
large volume (say 128Mpc/h x 128 Mpc/h). We do not the masking and long path 
lengths required for the covariance estimation, but do need to save into 
a format [RA, Dec] etc ingestible by the lyafxcorr_kg class.

We use Alex Krolewski's mock spectra, which is already noisy + continuum errors.
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

from astropy.cosmology import FlatLambdaCDM, z_at_value

from fnmatch import fnmatch

np.random.seed(5550423)

outname='mocks/pixel_radecz_mock_Proto128.bin'
xlow_tmp = 0.
ylow_tmp = 0.
xhi_tmp = 128.
yhi_tmp = 128.

# Also manually fix the number of skewers to something we expect
# Since 128^2 is roughly 20x the CLAMATO volume, we pick 240x20~5000 sightlines
nskewer=5000


# Define cosmology. To set the conversion between [RA,Dec] and [x,y], 
# we assume <z>=2.3
cosmo = FlatLambdaCDM(H0=70, Om0=0.31)
zmin = 2.0
zmid = 2.3
comdist_mean = cosmo.comoving_distance(zmid)
comdist_zmin = cosmo.comoving_distance(zmin)
dcomdist_dz = cosmo.inv_efunc(zmid) *2998. # in Mpc/h

#Read in the skewers, which are in 5-dimensional arrays of 
# [x,y,z,noise,delta_F]
skewers_orig = np.fromfile('mocks/allskewers_512cubed_46058nskew.bin')
skewers = np.reshape(skewers_orig, (int(np.shape(skewers_orig)[0]/5), 5))
npixels = int(np.shape(skewers_orig)[0]/5)

# Separate the pixels into individual skewers
xpos_pix= skewers[:,0]
ypos_pix= skewers[:,1]
zpos_pix= skewers[:,2]
sigma_pix = skewers[:,3]
flux_pix = skewers[:,4]

xy_pix = np.transpose([xpos_pix, ypos_pix])
pos_arr = [tuple(xytmp) for xytmp in xy_pix]
xy_uniq_arr = np.unique(pos_arr, axis=0)

xy_uniq = [tuple(xytmp) for xytmp in xy_uniq_arr]

n_skewer = len(xy_uniq)
npix_perskewer = np.int(npixels/ n_skewer)

print('{0} noisy skewers being processed'.format(n_skewer))

xskewer = np.asarray([tuptmp[0] for tuptmp in xy_uniq])
yskewer = np.asarray([tuptmp[1] for tuptmp in xy_uniq])

#Assign RA and Dec corresponding to x and y
ra_skewer = 180./np.pi  * xskewer/comdist_mean.value/cosmo.h
dec_skewer = 180./np.pi * yskewer/comdist_mean.value/cosmo.h

radec_uniq = list(zip(ra_skewer, dec_skewer))

coords = list(zip(xskewer, yskewer, ra_skewer, dec_skewer))

ra_vec = [] #np.asarray([None])
dec_vec = [] #np.asarray([None])
z_vec = [] #np.asarray([None])
sig_vec = [] #np.asarray([None])
f_vec = []

getskewers = np.all(np.column_stack([(xskewer > xlow_tmp), 
                                     (xskewer <= xhi_tmp),
                                     (yskewer > ylow_tmp), 
                                     yskewer <= yhi_tmp]), axis=1)

getskew_all= np.squeeze(np.nonzero(getskewers))

getskew_ind = np.random.choice(getskew_all, nskewer, replace=False)

coord_tmp1 = zip(xskewer[getskew_ind], yskewer[getskew_ind], 
                 ra_skewer[getskew_ind], dec_skewer[getskew_ind])

ctr = 0
for xtmp, ytmp, ra_tmp, dec_tmp in coord_tmp1:
    print(ctr, xtmp, ytmp)
    skewer_mask = np.all(np.column_stack([np.isclose(xtmp, xpos_pix,rtol=1.e-5),
                                          np.isclose(ytmp, ypos_pix, rtol=1.e-5)]),axis=1)

    flux = flux_pix[skewer_mask]
    sigma = sigma_pix[skewer_mask]
    
    if ctr == 0:
        zskewer = zpos_pix[skewer_mask]
        redshift_vec = zmin + zskewer/dcomdist_dz 
        red_tmp = np.ndarray.tolist(redshift_vec)

    ra_vec_tmp  = npix_perskewer * [np.asscalar(ra_tmp)]
    dec_vec_tmp = npix_perskewer * [np.asscalar(dec_tmp)]
    
    flux_tmp = np.ndarray.tolist(flux)
    sigma_tmp = np.ndarray.tolist(sigma)
    
    ra_vec.extend(ra_vec_tmp)
    dec_vec.extend(dec_vec_tmp)
    z_vec.extend(red_tmp)
    sig_vec.extend(sigma_tmp)
    f_vec.extend(flux_tmp)
    ctr+=1

npix_thismock = np.asarray(len(ra_vec), dtype='i4')
print("Saving {} pixels".format(npix_thismock))
ra_sav = np.asarray(ra_vec,dtype='f8')
dec_sav = np.asarray(dec_vec,dtype='f8')
z_sav  = np.asarray(z_vec,dtype='f8')
sig_sav = np.asarray(sig_vec,dtype='f8')
f_sav = np.asarray(f_vec,dtype='f8')

outfil = open(outname, 'w')

npix_thismock.tofile(outfil)
ra_sav.tofile(outfil)
dec_sav.tofile(outfil)
z_sav.tofile(outfil)
sig_sav.tofile(outfil)
f_sav.tofile(outfil)

outfil.close()

exit()
