import numpy as np
from astropy.io import ascii

def rebin_model_to_xcorr(delta_z, SigModel, PiModel, SigEdges, PiEdges):
    """Returns an array of subscripts to map model grid to observed grid, as a function of delta_z.
    
    Inputs: 
        delta_z   - Desired offset along LOS (pi) direction, in same units as Pi and Sigma (i.e. Mpc/h)
        SigModel, PiModel - sigma and pi positions of the models grid, as 1D vectors. This should 
                            the full vector of all pixels, not just unique val;ues
        SigEdges, PiEdges - sigma and pi bin edges of the output grid. Provide unique values 
        
    
    For each bin in the observed cross-correlation grid return a list of model grid 1D subscripts"""
    
    nbins_out = (len(SigEdges)-1) * (len(PiEdges)-1)
    
    rebinned_indices = [None] * nbins_out
    
    SigEdges1 = SigEdges[:-1]
    SigEdges2 = SigEdges[1:]
    PiEdges1  = PiEdges[:-1]
    PiEdges2  = PiEdges[1:]
    
    ctr = 0
    for p1, p2 in zip(PiEdges1,PiEdges2):
        for s1, s2 in zip(SigEdges1, SigEdges2):
            grabpix =  np.where( (SigModel >= s1) & (SigModel < s2) & 
                                (PiModel+delta_z >= p1) & (PiModel+delta_z < p2) )
            rebinned_indices[ctr] = grabpix
            ctr += 1

    return rebinned_indices

def xcorr_model_binned(XCorrModelFil, SigEdges, PiEdges, delta_z = 0., LinearModel=False):
    """Given filename of forest cross-correlation model as provided by model, 
    return 2D x-correlation array that could be directly compared with the CLAMATO XCorr arrays 
    and covariances as read in from the numpy files
    
    This is useful for comparing a single model 
    
    Here, we assume everything is in Mpc/h (i.e. the distance units in Andreu Font's models)
    
    Inputs: 
        XCOrrModelFil  - Model file name
        SigEdges       - 1D vector of output transverse bin edges (unique)
        PiEdges        - 1D vector of output LOS bin edges (unique)
        delta_z        - LOS offset, in Mpc/h
    
    For the 2017 CLAMATO X-Corr analysis, the distances in the model files are all Mpc/h 
    so the bin parameters input to this function should be as well.
    
    Returns by default the model convolved with Gaussian redshift space distortions. 
    'Linear' can be set to True to return the underlying linear model.
    """
    
    nbins_Pi = len(PiEdges)-1
    nbins_Sig = len(SigEdges)-1
    nbins_out = ( nbins_Pi * nbins_Sig)
    
    mod_raw = ascii.read(XCorrModelFil, format='no_header', 
                         names=['rt', 'rp', 'xi_lin', 'xi_conv'])

    rt = mod_raw['rt']
    rp = mod_raw['rp']
    xi_lin = mod_raw['xi_lin']
    xi_conv = mod_raw['xi_conv']

    # reflect along pi axis
    rp2 = rp * -1.

    # Create an structured array
    dtype =[('rp', 'f8'), ('rt', 'f8'), ('xi_conv', 'f8'), ('xi_lin', 'f8')]

    mod_raw = np.zeros(len(xi_conv)*2, dtype=dtype)

    mod_raw['rp'] = np.append(rp, rp2)
    mod_raw['rt'] = np.append(rt,rt)
    mod_raw['xi_conv'] = np.append(xi_conv, xi_conv)
    mod_raw['xi_lin'] = np.append(xi_lin, xi_lin)
    
    mod_raw = np.sort(mod_raw, order=['rt', 'rp'])
    
    if rebin_indices is None:
        rebinned_ind = rebin_model_to_xcorr(delta_z, mod_raw['rt'], mod_raw['rp'], 
                                            SigEdges, PiEdges)
    else:
        rebinned_ind = rebin_indices
    
    XCorr_model_flat = np.empty(nbins_out)
    
    if LinearModel:
        xi_flat = mod_raw['xi_lin'] 
    else:
        xi_flat = mod_raw['xi_conv']

    for i in range(nbins_out):
        ind_tmp = rebinned_ind[i][0]
        XCorr_model_flat[i] = np.mean(xi_flat[ind_tmp])
        
    XCorr_model = np.reshape(XCorr_model_flat, [nbins_Pi,nbins_Sig])
    return np.transpose(XCorr_model)

class XCorrModel:
    def __init__(self, ModFil):
        """ This is the class of Forest-Galaxy Cross-correlation models primarily for I/O.
        It will read in the file as a numpy structured array, and reflect the axes in the LOS dimension.
        """
        
        mod_raw = ascii.read(ModFil, format='no_header', 
                             names=['rt', 'rp', 'xi_lin', 'xi_conv'])

        rt = mod_raw['rt']
        rp = mod_raw['rp']
        xi_lin = mod_raw['xi_lin']
        xi_conv = mod_raw['xi_conv']

        # reflect along pi axis
        rp2 = rp * -1.

        # Create an structured array
        dtype =[('rp', 'f8'), ('rt', 'f8'), ('xi_conv', 'f8'), ('xi_lin', 'f8')]

        mod_refl = np.zeros(len(xi_conv)*2, dtype=dtype)

        mod_refl['rp'] = np.append(rp, rp2)
        mod_refl['rt'] = np.append(rt,rt)
        mod_refl['xi_conv'] = np.append(xi_conv, xi_conv)
        mod_refl['xi_lin'] = np.append(xi_lin, xi_lin)
    
        mod_refl = np.sort(mod_refl, order=['rt', 'rp'])

        self.rt = mod_refl['rt']
        self.rp = mod_refl['rp']
        self.xi_lin  = mod_refl['xi_lin']
        self.xi_conv = mod_refl['xi_conv']

    def rebin_flat(self, indices, LinearModel = False):
        """ Rebin the model into another grid. Input is a list of indices referencing
        the input model binning. 
        
        The input indices should be a list where each element is an array of indices, as 
        output by the function rebin_model_to_xcorr.
        """
        nbins_out = len(indices)
        model_flat = np.empty(nbins_out)

        if LinearModel:
            xi = self.xi_lin
        else:
            xi = self.xi_conv

        for i in range(nbins_out):
            ind_tmp = indices[i][0]
            model_flat[i] = np.mean(xi[ind_tmp])

        return model_flat

    def rebin2d(self,indices, SigEdges, PiEdges, LinearModel=False):
        """ Rebin the model into the 2D outgrid defined by SigEdges and PiEdges
        (transverse and LOS directions, respectively).

        Basically the same as REBIN_FLAT, but reshapes it and transposes it to a form
        where it can be directly compared with the data x-corr
        """
        flatmodel = self.rebin_flat(indices, LinearModel=LinearModel)

        model2d = np.reshape(flatmodel, [len(PiEdges)-1, len(SigEdges)-1])

        return np.transpose(model2d)
