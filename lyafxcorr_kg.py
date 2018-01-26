import numpy as np
import time as time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

cosmo_default = FlatLambdaCDM(H0=70,Om0=0.31)

class lyapix:   
    def __init__(self, path, cosmo=cosmo_default):
        """
        This class represents Ly-a forest pixel positions, delta_F, and noise of Ly-a forest pixels 
        from 2016 CLAMATO data.

        Also defines pixel weights,SkyCoord variables and LOS distance for the class.
        """
        
        # get file handle
        pixfile = open(path, "r")

        # use fromfile to read binary chunks.
        # First number in the file is the number of pixels
        npix = np.fromfile(pixfile, dtype='i4', count=1)
        self.npix = np.asscalar(npix)

        # Read in quantities directly stored in the binary file
        self.ra = np.fromfile(pixfile, dtype='f8',count=self.npix)
        self.dec = np.fromfile(pixfile, dtype='f8', count=self.npix)
        self.z = np.fromfile(pixfile, dtype='f8', count=self.npix)
        self.sig = np.fromfile(pixfile, dtype='f8', count=self.npix)
        self.delta = np.fromfile(pixfile, dtype='f8',count=self.npix)

        #done reading
        pixfile.close()

        # compute pixel weights
        self.w = 1./(0.065*((1.+self.z)/3.25)**3.8 + self.sig**2)

        # Compute comoving LOS distance to each pixel
        self.comdist = cosmo.comoving_distance(self.z)
        
        self.coord = SkyCoord(ra = self.ra*u.degree, dec=self.dec*u.degree,
                              distance=self.comdist)
    def var_forest(z_in):
        """ Return intrinsic Lya-forest variance as function of redshift"""
        return 0.065 * ( (1.+z_in)/3.25 )**3.8

    def resample(self):
        """
        Randomly resample (with replacement) all the forest pixels of 
        the instance. This is done in-place, so should only operate on 
        *copies* of the true pixel distribution.

        Also creates a new attribute called ind_vec, which is the 
        sequence of indices used to create the current resample.
        """
        ind_vec = np.random.choice(self.npix, self.npix, replace=True)

        dectmp = self.dec
        ztmp = self.z
        sigtmp = self.sig
        deltatmp = self.delta
        wtmp = self.w
        comdisttmp = self.comdist
        coordtmp = self.coord
        
        self.ra = self.ra[ind_vec]
        self.dec = dectmp[ind_vec]
        self.z   = ztmp[ind_vec]
        self.sig = sigtmp[ind_vec]
        self.delta = deltatmp[ind_vec]
        self.w     = wtmp[ind_vec]
        self.comdist = comdisttmp[ind_vec]
        self.coord   = coordtmp[ind_vec]

        self.ind_vec = ind_vec
        return self

    def resample_skewer(self, SkewerRec=None):
        """
        Randomly resample (with replacement) the forest sightlines of 
        the instance. This is done in-place, so should only operate on 
        *copies* of the true pixel distribution.

        Also creates a new attribute called ind_vec, which is the 
        sequence of indices used to create the current resample.

        SkewerRec is a record array listing the sightline pixels that can be 
        optionally given as input created during each call. Otherwise, gen_SkewerRec
        will be run.
        """
        if SkewerRec == None:
            SkewerRec = self.gen_SkewerRec()

        SkewerPos, SkewerMask = SkewerRec

        # Convert SkewerMask into an array
        SkewerMask = np.asarray(SkewerMask)
        
        n_skewer = len(SkewerPos)
        
        # Resample the skewers with replacement. This returns skewer indices
        rand_skewer = np.random.choice(n_skewer, n_skewer, replace=True)

        # Convert the skewer masks into an array
        SkewerMask_rand = SkewerMask[rand_skewer,:]

        # Collect the indices into a single array
        ind_vec = []

        for skewer in SkewerMask_rand:
            if len(ind_vec)==0:
                ind_vec = np.where(skewer)
            else:
                ind_vec = np.append(ind_vec,np.where(skewer))

        dectmp = self.dec
        ztmp = self.z
        sigtmp = self.sig
        deltatmp = self.delta
        wtmp = self.w
        comdisttmp = self.comdist
        coordtmp = self.coord
        
        self.ra = self.ra[ind_vec]
        self.dec = dectmp[ind_vec]
        self.z   = ztmp[ind_vec]
        self.sig = sigtmp[ind_vec]
        self.delta = deltatmp[ind_vec]
        self.w     = wtmp[ind_vec]
        self.comdist = comdisttmp[ind_vec]
        self.coord   = coordtmp[ind_vec]

        self.ind_vec = ind_vec
        return self

    def gen_SkewerRec(self):
        """ Create a tuple with (1) array of the individual skewers' [RA/Dec]
        and (2) pixel indices referencing the overall class instance
        """

        # Create tuples of [ra,dec] then find unique row
        radec_all = np.transpose([self.ra, self.dec])
        pos_arr = [tuple(radectmp) for radectmp in radec_all]
        radec_uniq_arr = np.unique(pos_arr,axis=0)

        radec_uniq = [tuple(postmp) for postmp in radec_uniq_arr]

        n_skewer = len(radec_uniq)

        dtype = [('RA','f8'), ('Dec', 'f8')]

        skewer_pos = np.empty(n_skewer, dtype=dtype)
        
        skewer_mask = [None] * n_skewer

        ctr = 0
        for ra_tmp, dec_tmp in radec_uniq:
            skewer_pos['RA'][ctr] = ra_tmp
            skewer_pos['Dec'][ctr] = dec_tmp
            skewer_mask[ctr] = np.all(np.column_stack( 
                [np.isclose(ra_tmp, self.ra,rtol=1.e-5),
                 np.isclose(dec_tmp, self.dec, rtol=1.e-5)]), axis=1)
            ctr +=1

        SkewerRec = (skewer_pos, skewer_mask)

        return SkewerRec
        

def countpix(GalCoord, PixCoord,  SigEdges, PiEdges,
             cosmo=cosmo_default):
    """ Given a galaxy position and redshift, returns a 2D histogram 
    of Lya-forest pixels in distance bins, with the bin edges 
    defined by PiEdges and SigEdges. The LOS comoving distance should 
    already be stored in the SkyCoord objects.

    Also takes an astropy cosmology object for conversion between 
    redshift and comoving distance
    """
    
    # Evaluate LOS comoving separation between galaxy and forest pixels.
    LosSep = PixCoord.distance - GalCoord.distance

    # This is the mean redshift between galaxy and each pixel, to
    # evaluate transverse comoving distances
    # Jul 15: Just directly get mean of comoving distance
    #zMeanTmp = (zPix + np.repeat(zGal,npix)) / np.repeat(2., npix) 

    # Discard pixels that are greater in LOS distance than the desired
    # boundaries. This is a quick cut to ease histogram evaluation
    PiBound = (min(PiEdges), max(PiEdges))
    get_near = np.all(np.column_stack([(LosSep >= PiBound[0]*u.Mpc),
                                      (LosSep <= PiBound[1]*u.Mpc)]),
                     axis=1)

    PixCoordTmp = PixCoord[get_near]
    LosSep = LosSep[get_near]
    n_near = len(PixCoordTmp)
    #zMeanTmp = zMeanTmp[get_near]
    ComDistMean = (PixCoordTmp.distance + np.repeat(GalCoord.distance,n_near))/ \
                  np.repeat(2., n_near)

    AngleSep = PixCoordTmp.separation(GalCoord)
    TransSep = AngleSep.radian * ComDistMean

    Hist2D, xedges, yedges = np.histogram2d(TransSep, LosSep,
                                            bins=[SigEdges, PiEdges])
    
    return Hist2D

def weight_stack(GalCoord, PixCoord, WeightsIn, SigEdges, PiEdges,
             cosmo=cosmo_default):
    """ Given a galaxy position and redshift, returns a weighted 2D histogram 
    of Lya-forest pixels in distance bins, with the bin edges 
    defined by PiEdges and SigEdges. The LOS comoving distance should 
    already be stored in the SkyCoord objects.

    Also takes an astropy cosmology object for conversion between 
    redshift and comoving distance
    """
    
    # Evaluate LOS comoving separation between galaxy and forest pixels.
    LosSep = PixCoord.distance - GalCoord.distance

    # This is the mean redshift between galaxy and each pixel, to
    # evaluate transverse comoving distances
    # Jul 15: Commented out. We'll just directly get mean of comoving distance
    #zMeanTmp = (zPix + np.repeat(zGal,npix)) / np.repeat(2., npix) 

    # Discard pixels that are greater in LOS distance than the desired
    # boundaries. This is a quick cut to ease histogram evaluation
    PiBound = (min(PiEdges), max(PiEdges))
    get_near = np.all(np.column_stack([(LosSep >= PiBound[0]*u.Mpc),
                                      (LosSep <= PiBound[1]*u.Mpc)]),
                     axis=1)

    PixCoordTmp = PixCoord[get_near]
    LosSep = LosSep[get_near]
    Weights = WeightsIn[get_near]
    n_near = len(PixCoordTmp)
    #zMeanTmp = zMeanTmp[get_near]
    ComDistMean = (PixCoordTmp.distance + np.repeat(GalCoord.distance,n_near))/ \
                  np.repeat(2., n_near)

    AngleSep = PixCoordTmp.separation(GalCoord)
    TransSep = AngleSep.radian * ComDistMean

    WHist2D, xedges, yedges = np.histogram2d(TransSep, LosSep, weights=Weights,
                                            bins=[SigEdges, PiEdges])
    
    return WHist2D

def xcorr_gal_lya(GalCoord, LyaPix, SigEdges, PiEdges, cosmo=cosmo_default,verbose=1):
    """ 
    Perform cross-correlation between a set of galaxy SkyCoord positions and a list of
    Lya-forest pixels (in a lyapix object). Also need to define the cross-correlation
    bins

    Return a tuple with (XCorr_arr, NoXCorr_arr), where NoXCorr_arr is an array listing 
    galaxy indices that had no cross-correlations within the defined bins.
    """

    if verbose == 1:
        print("Evaluating cross-correlation for %i galaxies and %i forest pixels" % (len(GalCoord), LyaPix.npix)) 
    
    # Stack the first object to initialize the output array
    itmp = 0
    CoordTmp = GalCoord
    
    t0= time.clock()
    NumerArr = weight_stack(CoordTmp[itmp], LyaPix.coord, LyaPix.delta*LyaPix.w, SigEdges, 
                                  PiEdges, cosmo=cosmo)
    DenomArr = weight_stack(CoordTmp[itmp], LyaPix.coord, LyaPix.w, SigEdges, 
                                  PiEdges, cosmo=cosmo)
    if verbose == 1:
        print('2D histogram evaluation took %f seconds for first galaxy' % (time.clock()-t0))

    NoNearPix = []
    for itmp in range(1,len(GalCoord)):
        CoordTmpHere = CoordTmp[itmp]
        NumerTmp = weight_stack(CoordTmpHere, LyaPix.coord, LyaPix.delta*LyaPix.w, 
                                      SigEdges, PiEdges, cosmo=cosmo)
        DenomTmp = weight_stack(CoordTmpHere, LyaPix.coord, LyaPix.w,
                                      SigEdges, PiEdges, cosmo=cosmo)
        NumerArr += NumerTmp
        DenomArr += DenomTmp
        if np.isclose(np.sum(DenomTmp),0., atol=1.e-6): 
                NoNearPix=NoNearPix.append(itmp)
        
    if verbose == 1:
        print("Finished evaluating cross-correlations. This took %f seconds"
              % (time.clock()-t0) )
        print("%i galaxies had no cross-correlations within these bins." % len(NoNearPix))

    CrossCorr = NumerArr * (DenomArr != 0)/ (DenomArr + (DenomArr == 0))
    return (CrossCorr, NoNearPix)
