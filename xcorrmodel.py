import numpy as np
from astropy.io import ascii
import re
from scipy import interpolate

######## A couple of functions to replace multiple strings #########
def multiple_replacer(*key_values):
    replace_dict = dict(key_values)
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values]), re.M)
    return lambda string: pattern.sub(replacement_function, string)

def multiple_replace(string, *key_values):
    return multiple_replacer(*key_values)(string)
###################################################################

def rebin_model_to_xcorr(delta_z, SigModel, PiModel, SigEdges, PiEdges,verbose=False):
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
            if verbose: 
                print(ctr, len(grabpix[0]))
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
    

    rebinned_ind = rebin_model_to_xcorr(delta_z, mod_raw['rt'], mod_raw['rp'], 
                                            SigEdges, PiEdges)
    
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

        self.nbin_raw = len(xi_conv)*2
        self.rt = mod_refl['rt']
        self.rp = mod_refl['rp']
        self.xi_lin  = mod_refl['xi_lin']
        self.xi_conv = mod_refl['xi_conv']
        # Store also the model filename, and parse the parameter values from it
        # This is for v0 models, and might need to change for future versions
        self.modelfil = ModFil
        self.bias  = float(ModFil.split("_")[-2].replace("b",""))

        sig_repl = (u"s", u""), (u".txt", u"")
        self.sig_z = float(multiple_replace(ModFil.split("_")[-1], *sig_repl))

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

    def rebin_from_dz(self, SigEdges, PiEdges, delta_z=0., 
                       LinearModel=False):
        """ Same functionality as the xcorr_model_binned method, but operates on 
        the object rather than reading from file."""
        print("This method hasn't been tested at all!!")
        nbins_Pi = len(PiEdges)-1
        nbins_Sig = len(SigEdges)-1
        nbins_out = ( nbins_Pi * nbins_Sig)

        rebinned_ind = rebin_model_to_xcorr(delta_z, self.rt, self.rp, 
                                            SigEdges, PiEdges)
        XCorr_model_flat = np.empty(nbins_out)
    
        if LinearModel:
            xi_flat = self.xi_lin 
        else:
            xi_flat = self.xi_conv

        for i in range(nbins_out):
            ind_tmp = rebinned_ind[i][0]
            XCorr_model_flat[i] = np.mean(xi_flat[ind_tmp])
        
            XCorr_model = np.reshape(XCorr_model_flat, [nbins_Pi,nbins_Sig])
            return np.transpose(XCorr_model)

# UNDER CONSTRUCTION
class ModelFunc:
    def __init__(self, ModArr):
        """
        Class for calling cross-correlation models as a function of input parameter. 
        This currently works for models that define a uniform grid in b and sigz, in 
        preparation for interpolation.

        Unlike XCorrModel, we do NOT store the reflected xi both towards and away from LOS. 
        This is to save memory, as we will need to store a large grid of models in memory.
        
        This is initialized by passing a list of XCorrModels. 
        """
        self.nmodel = np.size(ModArr)
        
        bias_arr = np.empty(self.nmodel)
        sigz_arr = np.empty(self.nmodel)

        nbin_raw = ModArr[1].nbin_raw
        self.nbin = int(nbin_raw/2)
        rt2 = ModArr[1].rt
        rp2 = ModArr[1].rp

        mask_neg = rp2 >= 0.

        self.rt = rt2[mask_neg]
        self.rp = rp2[mask_neg]

        # First loop through model arr to get all the parameters
        ctr = 0
        for mod in ModArr:
            bias_arr[ctr] = mod.bias
            sigz_arr[ctr] = mod.sig_z
            ctr += 1

        self.bias_arr = bias_arr
        self.sigz_arr = sigz_arr
        #self.xi_conv = xi_conv_arr

        # Create arrays of unique parameter values
        self.bias = np.unique(bias_arr)
        self.sigz = np.unique(sigz_arr)
        self.nbias = len(self.bias)
        self.nsigz = len(self.sigz)

        # Define the dictionary to map between parameter values and array indices.
        # We round the key values to ensure a stable hash
        bias_dic = dict(zip(np.round(self.bias,1), np.arange(self.nbias)))
        sigz_dic = dict(zip(np.round(self.sigz,5), np.arange(self.nsigz)))

        # Initialize array for the xi, with two additional dimensions for the parameters. 
        # Fill with NaNs so that we can check at the end that every entry is filled
        xi_conv_2d = np.empty((len(self.bias), len(self.sigz), self.nbin))
        xi_conv_2d[:] = np.nan

        for mod in ModArr:
            i_b = bias_dic[np.round(mod.bias,1)]
            i_s = sigz_dic[np.round(mod.sig_z,5)]
            xi_conv_2d[i_b,i_s,:] = mod.xi_conv[mask_neg]

        if np.isnan(xi_conv_2d).any():
            print("Error: Parameter space not uniformly sampled. Need a model for every (b, sig_z)")
            
        self.xi_conv = xi_conv_2d
        
    def XCorrOut(self,bias_in, sigz_in):
        """ Return raw 2D cross-correlation model given bias and sigma_z that is exactly part of the model grid."""
        # Check that the desired bias and sig_z values are part of the current object's parameter grid
        bias_dic = dict(zip(np.round(self.bias,1), np.arange(self.nbias)))
        sigz_dic = dict(zip(np.round(self.sigz,5), np.arange(self.nsigz)))

        if not np.round(bias_in,1) in bias_dic.keys():
            print("Error: input bias value not part of parameter grid")
            print("Available bias values:")
            print(self.bias)
            return

        if not np.round(sigz_in,5) in sigz_dic.keys():
            print("Error: input sig_z value not part of parameter grid")
            print("Available sig_z values:")
            print(self.bias)
            return

        i_b = bias_dic[np.round(bias_in,1)]
        i_s = sigz_dic[np.round(sigz_in,5)]

        return np.squeeze(self.xi_conv[i_b,i_s,:])

    def XCorrInterpRaw(self, bias_in, sigz_in):
        """ Interpolate the xi from the ModelFunc object, given an input bias and sig_z
        """
        if bias_in < np.min(self.bias) or bias_in > np.max(self.bias):
            print("Warning: input bias value outside parameter space. We are extrapolating...")

        if sigz_in < np.min(self.sigz) or sigz_in > np.max(self.sigz):
            print("Warning: input sigz value outside parameter space. We are extrapolating...")

        xi_out = np.empty(self.nbin)

        for ibin in np.arange(self.nbin):
            #if (ibin % 1000) == 0:
            #    print(ibin)
            xi_arr_tmp = np.squeeze(self.xi_conv[:,:,ibin])
            xi_out[ibin] = interpolate.interpn( (self.bias, self.sigz), xi_arr_tmp, [bias_in, sigz_in],bounds_error=False,fill_value=None)
            
        return xi_out

    def XCorrInterpBin(self,  bias_in, sigz_in, dz_in, SigEdges, PiEdges): 
            # 1. generate reflected LOS axis
            rt2 = np.append(self.rt, self.rt)
            rp2 = np.append(self.rp, self.rp * -1.)
            
            xi_tmp = self.XCorrInterpRaw(bias_in, sigz_in)
            xi2 = np.append(xi_tmp, xi_tmp)

            bin_ind = rebin_model_to_xcorr(dz_in, rt2, rp2, SigEdges, PiEdges, verbose=False)

            nbins_Pi = len(PiEdges)-1
            nbins_Sig = len(SigEdges)-1
            nbins_out = ( nbins_Pi * nbins_Sig)

            XCorrBinned_flat = np.empty(nbins_out)

            for i in range(nbins_out):
                ind_tmp = bin_ind[i][0]
                XCorrBinned_flat[i] = np.mean(xi2[ind_tmp])

            XCorrBinned = np.reshape(XCorrBinned_flat, [nbins_Pi, nbins_Sig])
            return np.transpose(XCorrBinned)

