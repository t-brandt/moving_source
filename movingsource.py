import numpy as np
from scipy import special, interpolate, optimize
import fitramp_movingsource

import warnings

class Covar:

    """
    Convenience class for covariance matrix information.

    """
    
    def __init__(self, read_times, integrated_counts=None):

        """

        Compute components of the (tridiagonal) covariance matrix.

        There will be three components: photon noise from a static
        scene, read noise, and photon noise from a moving source with a
        known illumination as a function of time.

        Inputs
        ------
        readtimes : list
             List of values or lists for the times of reads.  If a list of
             lists, times for reads that are averaged together to produce
             a resultant.
        integrated_counts : list of ndarrays or None
             Each ndarray is of dimension n_reads x npixels.  Expected
             counts in each resultant at each pixel for a moving source
             of unit flux.  If None, only compute the static illumination
             terms of the covariance matrix.
             
        Computes the components of the (tridiagonal) covariance matrix.
        These are:
        
        alpha_phnoise: diagonal components for the static scene
        beta_phnoise: off-diagonal components for the static scene
        alpha_readnoise: diagonal components for the read noise
        beta_readnoise: off-diagonal components for the read noise

        Also computed if integrated_counts is not None:
        alpha_phnoise_path: diagonal components for a moving object of
            unit flux
        beta_phnoise_path: off-diagonal components for a moving object of
            unit flux
        
        Note that alpha_phnoise_path and beta_phnoise_path both have an
        extra dimension over the other arrays, as they have different
        values for each pixel.
        
        """

        mean_t = []   # mean time of the resultant as defined in the paper
        tau = []      # variance-weighted mean time of the resultant

        mean_t_path = []
        tau_path = []
        _N = []
        
        N = []  # Number of reads per resultant
        
        for times in read_times:
            mean_t += [np.mean(times)]
            
            if hasattr(times, "__len__"):
                N += [len(times)]
                k = np.arange(1, N[-1] + 1)
                tau += [1/N[-1]**2*np.sum((2*N[-1] + 1 - 2*k)*np.array(times))]
            else:
                tau += [times]
                N += [1]

        if integrated_counts is not None:
            # This is treated the same as photon noise for a static scene,
            # but 
            for counts in integrated_counts:
                mean_t_path += [np.mean(counts, axis=0)]
                
                if hasattr(counts, "__len__"):
                    Ncts = len(counts)
                    k = np.arange(1, Ncts + 1)[:, None]
                    tau_path += [1/Ncts**2*np.sum((2*Ncts + 1 - 2*k)*
                                                  (np.array(counts)), axis=0)]
                else:
                    tau_path += [np.array(counts)]

        mean_t = np.array(mean_t)
        tau = np.array(tau)
        N = np.array(N)
        
        delta_t = mean_t[1:] - mean_t[:-1]

        self.delta_t = delta_t
        self.mean_t = mean_t
        self.tau = tau
        self.Nreads = N
        
        self.alpha_readnoise = (1/N[:-1] + 1/N[1:])/delta_t**2
        self.beta_readnoise = -1/N[1:-1]/(delta_t[1:]*delta_t[:-1])
        
        self.alpha_phnoise = (tau[:-1] + tau[1:] - 2*mean_t[:-1])/delta_t**2
        self.beta_phnoise = (mean_t[1:-1] - tau[1:-1])/(delta_t[1:]*delta_t[:-1])
        if integrated_counts is not None:
            mean_t_path = np.array(mean_t_path)
            tau_path = np.array(tau_path)
        
            self.alpha_phnoise_path = (tau_path[:-1] + tau_path[1:] - 2*mean_t_path[:-1])/delta_t[:, None]**2
            self.beta_phnoise_path = (mean_t_path[1:-1] - tau_path[1:-1])/(delta_t[1:]*delta_t[:-1])[:, None]


def bin_to_resultants(fullarray, resultants, for_cov=False):
    """
    Take an array of reads and make a list or array of resultants.

    Inputs
    ------
    fullarray : ndarray
         Signal at each read, dimensions nreads x npixels.
    resultants : list
         Either a list of numbers or a list of lists, where the list
         gives the indices of the reads that comprise the resultant.
    for_cov : bool, optional
         Return a list for the covariance matrix calculation?  If
         False, return an array for the actual resultant values.
         Default False
    
    Returns
    -------
    output : ndarray or list
         Either return a list of arrays, where each array in the list
         is 2D where the first dimension is the number of reads
         comprising that resultant, or return a 2D array of values for
         the resultants computed from averaging the constituent reads.
    
    """

    output = []

    for resultant in resultants:
        if isinstance(resultant, list):
            if for_cov:
                output += [np.array([fullarray[i] for i in resultant])]
            else:
                output += [np.mean([fullarray[i] for i in resultant], axis=0)]
        else:
            if for_cov:
                output += [np.array(fullarray[resultant])[None, :]]
            else:
                output += [np.array(fullarray[resultant])]
    if for_cov:
        return output
    else:
        return np.array(output)


def rotate_image(image, theta):
    """
    Rotate an image using RectBivariateSpline to interpolate.

    Inputs
    ------
    image : ndarray
        The two dimensional ndarray to be rotated.
    theta : float
        The angle (in radians) by which to rotate counterclockwise.

    Returns
    -------
    rotated : ndarray
        The rotated two-dimensional ndarray.  Will be the same size as
        the input array.
    """

    # Construct coordinates for the rotated image.

    x, y = np.indices(image.shape)
    xcen, ycen = np.mean(x), np.mean(y)
    x = x - xcen
    y = y - ycen

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Now rotate the image by theta, return the rotated image.

    xp = r*np.cos(phi + theta) + xcen
    yp = r*np.sin(phi + theta) + ycen

    f = interpolate.RectBivariateSpline(np.arange(image.shape[0]),
                                        np.arange(image.shape[1]), image)
    rotated = f(xp, yp, grid=False)

    return rotated


def compute_smeared_image(image, theta, distance):

    """
    Compute an image smeared out by constant motion.

    Inputs
    ------
    image : ndarray
        The two-dimensional, oversampled image in the absence of motion
    theta : float
        The angle in radians of the motion.  Measured clockwise with
        respect to the vertical axis.
    distance : float
        The distance in pixels over which to smear the image.

    Returns
    -------
    smeared : ndarray
        The two-dimensional image smeared by constant linear motion.
        Dimensions will match those of the input image.
    """

    # Compute the rotated image and its derivatives

    rotated = rotate_image(image, theta)
    y = np.arange(image.shape[0])
    x = np.arange(image.shape[1])

    f = interpolate.RectBivariateSpline(y, x, rotated)
    val = rotated
    d1 = f(y, x, dx=1)
    d2 = f(y, x, dx=2)
    d3 = np.zeros(d2.shape)
    d3[1:] = np.diff(d2, axis=0)

    imax = int(distance) + 1
    smeared = np.zeros(val.shape)

    # Use a Taylor expansion to get the exact subimage to add from the
    # value and derivatives of the bicubic spline representation.

    for i in range(imax):
        dy = -min(1, distance - i)
        if i == 0:
            smeared -= val*dy + d1/2*dy**2 + d2/6*dy**3 + d3/24*dy**4
        else:
            smeared[i:] -= val[:-i]*dy + d1[:-i]/2*dy**2 + d2[:-i]/6*dy**3 + d3[:-i]/24*dy**4

    # Normalize back to unit flux

    smeared /= distance

    return rotate_image(smeared, -theta)


def add_to_image(im, f, fshape, xpos, ypos, oversample=1):
    """
    Add a smeared subimage, represented by a 2D spline, to an image.

    Operates on im in place, returns None.
    
    Inputs
    ------
    im : ndarray
         The image into which to add smeared subimages
    f : scipy.interpolate.RectBivariateSpline
         A 2D bivariate spline representation of the smeared image to be
         added with a specified offset.  The center of the spline
         representation is assumed to be at fshape//2//oversample (see
         subsequent arguments)
    fshape : tuple
         The shape of the image represented by f
    xpos : float
         The x position (in pixels) at which to add the smeared subimage
    ypos : float
         The y position (in pixels) at which to add the smeared subimage
    oversample : int
         The factor by which pixels in f oversample pixels in im.

    Returns
    -------
    None
    
    """

    # Need to separate integer, fractional components of the offset
    # Fractional component will be handled by interpolation, integer
    # component will be handled by offsets of the subimage.

    imshape = im.shape
    xpos_int = int(xpos)
    ypos_int = int(ypos)
    ypos_frac = ypos - ypos_int
    xpos_frac = xpos - xpos_int

    # Assume the center is at shape//2//oversample.  These are the
    # integer positions relative to that center.

    i0 = ypos_int - fshape[0]//2//oversample
    j0 = xpos_int - fshape[1]//2//oversample

    # Fractional component for interpolation

    subim = f((np.arange(fshape[0]//oversample) - ypos_frac)*oversample, 
              (np.arange(fshape[1]//oversample) - xpos_frac)*oversample)

    di = max(im.shape[0], subim.shape[0])
    dj = max(im.shape[1], subim.shape[1])
    k0, l0 = 0, 0
    
    # Integer component handled by offsets in the inpainting.  Ensure
    # that we limit di (the size of the image to paint in) so that it
    # does not run off the end of either grid.

    if i0 < 0:
        di += i0
        k0 -= i0
        i0 = 0
    if i0 + di > imshape[0]:
        di = imshape[0] - i0

    if k0 < 0:
        di += k0
        i0 -= k0
        k0 = 0  
    if k0 + di > subim.shape[0]:
        di = subim.shape[0] - k0
        
    if j0 < 0:
        dj += j0
        l0 -= j0
        j0 = 0
    if j0 + dj > imshape[1]:
        dj = imshape[1] - j0

    if l0 < 0:
        dj += l0
        j0 -= l0
        l0 = 0          
    if l0 + dj > subim.shape[1]:
        dj = subim.shape[1] - l0

    # Assuming that there is a valid region to add on
    
    if dj > 0 and di > 0:
        im[i0:i0 + di, j0:j0 + dj] += subim[k0:k0 + di, l0:l0 + dj]
    else:
        raise ValueError("Attempting to add an image out of bounds")

    
def make_templates(
        epsf,
        phi,
        dist,
        x0,
        y0,
        readtimes,
        resultants,
        oversample,
        outshape,
        bigpix=False,
        threshold=1e-4,
        psf_saver=None,
        ipix=None
):
    
    """
    Construct a template of a moving source through a series of resultants.

    Inputs
    ------
    epsf : ndarray
        Two dimensional ndarray represented the effective PSF
    phi : float
        Angle in radians of the motion, measured clockwise with respect
        to the vertical axis
    dist : float
        Distance in pixels by which the source moves in one read
    x0 : float
        Starting position of the moving source in x
    y0 : float
        Starting position of the moving source in y
    readtimes : list
        The indices of all of the individual reads that are binned to
        construct the resultants
    resultants : list
        List of values or lists for the indices of reads.  If a list of
        lists, indices for reads that are averaged together to produce
        a resultant.
    oversample : int
        Factor by which epsf oversamples the output image
    outshape : tuple
        Shape of the output array in which to draw the moving source
    bigpix : bool, Optional
        Limit the calculation to pixels with a significant contribution
        from the moving source?  Default False
    threshold : float, Optional
        Fraction of peak pixel value to serve as the cutoff for keeping
        pixels.  Used only if bigpix is True.  Default 1e-4.
    psf_saver : PSF_interp_tool or None, optional
        Stores and updates the machinery to use a Taylor expansion to
        make a small change to the smeared ePSF.  Improves performance
        while this routine is optimized with many small steps.
    ipix : ndarray or None, optional
        Boolean array for which indices are actually computed with a
        full chi squared calculation

    Returns
    -------
    results : dict
        Dictionary with the following keys:

        "alltemplates" : ndarray
            integrated counts from a moving source read-by-read
        "templates_resultants" : ndarray
            alltemplates, binned into resultants
        "templates_resultants_cumsum" : list
            list of intra-resultant cumulative sums of the above
        "ipix" : ndarray
            boolean array that is True for pixels where the cumsum
            array listed immediately above exceeds threshold
    
    """

    # This block uses a structure to save the smeared ePSF and to use
    # a first-order Taylor expansion for very small changes in distance
    # or direction of motion.  Limit the Taylor expansion to a step of
    # 0.05 pixels in the supersampled ePSF.  If the step is larger than
    # this, compute a full new smeared ePSF and recompute the derivatives.

    if (
            psf_saver is not None and
            psf_saver.last_phi is not None and
            psf_saver.last_dist is not None and
            np.abs(psf_saver.last_phi - phi) < 5e-2/(dist*oversample) and
            np.abs(psf_saver.last_dist - dist) < 5e-2/oversample and
            psf_saver.last_smeared_epsf is not None
            ):

        im_smeared = psf_saver.last_smeared_epsf.copy()

        if psf_saver.dim_dphi is None or psf_saver.dim_ddist is None:
            ddist = 1e-3/oversample
            dim_ddist = compute_smeared_image(epsf, psf_saver.last_phi, (psf_saver.last_dist + ddist)*oversample)
            dim_ddist = 1/ddist*(dim_ddist[oversample:-oversample, oversample:-oversample] - im_smeared)
            psf_saver.dim_ddist = dim_ddist
            
            dphi = 1e-3*psf_saver.last_dist*oversample
            dim_dphi = compute_smeared_image(epsf, psf_saver.last_phi + dphi, psf_saver.last_dist*oversample)
            dim_dphi = 1/dphi*(dim_dphi[oversample:-oversample, oversample:-oversample] - im_smeared)
            psf_saver.dim_dphi = dim_dphi

        im_smeared += (phi - psf_saver.last_phi)*psf_saver.dim_dphi
        im_smeared += (dist - psf_saver.last_dist)*psf_saver.dim_ddist
    else:

        # Recompute the smeared ePSF

        im_smeared = compute_smeared_image(epsf, phi, dist*oversample)

        # Clip the image by one full pixel in all directions to avoid
        # artifacts from extrapolation.
    
        im_smeared = im_smeared[oversample:-oversample, oversample:-oversample]

        if psf_saver is not None:
            psf_saver.last_phi = phi
            psf_saver.last_dist = dist
            psf_saver.last_smeared_epsf = im_smeared
            psf_saver.dim_dphi = None
            psf_saver.dim_ddist = None

    # Represent the smeared ePSF as a spline so that it can be
    # interpolated.

    f2 = interpolate.RectBivariateSpline(np.arange(im_smeared.shape[0]),
                                         np.arange(im_smeared.shape[1]),
                                         im_smeared)

    nreads = len(readtimes)
    alltemplates = np.zeros((nreads, outshape[0], outshape[1]))
    for i in range(nreads):
        xpos, ypos = x0 + dist*i*np.sin(phi), y0 + dist*i*np.cos(phi)
        add_to_image(alltemplates[i], f2, im_smeared.shape, xpos, ypos, oversample=oversample)
    
    alltemplates[alltemplates <= 0] = 0
    alltemplates = alltemplates.reshape((nreads, -1))

    templates_cumsum = np.cumsum(alltemplates, axis=0)

    ipix_reshaped = None
    if bigpix and ipix is None:
        _ipix = templates_cumsum[-1] > threshold*np.amax(templates_cumsum[-1])
        ipix_reshaped = np.zeros(alltemplates.shape, dtype=bool)
        ipix_reshaped[:] = _ipix
    elif ipix is not None:
        ipix_reshaped = np.zeros(alltemplates.shape, dtype=bool)
        ipix_reshaped[:] = ipix
    if ipix_reshaped is not None:
        alltemplates = alltemplates[ipix_reshaped].reshape((nreads, -1))
        templates_cumsum = templates_cumsum[ipix_reshaped].reshape((nreads, -1))
        ipix_return = ipix_reshaped[0]
    else:
        ipix_return = None

    templates_resultants = np.diff(bin_to_resultants(templates_cumsum, resultants), axis=0)
        
    templates_resultants_cumsum = bin_to_resultants(templates_cumsum, resultants, for_cov=True)

    return {
        "alltemplates" : alltemplates,
        "templates_resultants" : templates_resultants,
        "templates_resultants_cumsum" : templates_resultants_cumsum,
        "ipix" : ipix_return
    }


def full_chisq(p,
               diffs,
               diffs2use,
               epsf,
               sig,
               readtimes,
               resultants,
               oversample,
               outshape,
               psf_saver=None,
               ipix_in=None,
               return_ancillary=False
               ):

    """
    Compute the total chi squared value for a ramp over many pixels.

    Intended to be passed to a nonlinear optimization routine.

    Inputs
    ------
    p : list of floats
        Four entries: phi, angle in radians of the source's motion
                      dist, motion per read in pixels
                      x0, x position of the moving source at t=0
                      y0, y position of the moving source at t=0
    
    diffs : ndarray
        Differences between consecutive reads, shape (nreads - 1, npixels).
    diffs2use : ndarray
        Boolean array of the same shape as diffs indicating which resultant
        differences should be used in the fit
    epsf : ndarray
        Two dimensional ndarray represented the effective PSF
    sig : ndarray
        Two dimensional ndarray of read noise
    readtimes : list
        The indices of all of the individual reads that are binned to
        construct the resultants
    resultants : list
        List of values or lists for the indices of reads.  If a list of
        lists, indices for reads that are averaged together to produce
        a resultant.
    oversample : int
        Factor by which epsf oversamples the output image
    outshape : tuple
        Shape of the output array in which to draw the moving source
    psf_saver : Chisq_SavedState
        Stores and updates the machinery to use a Taylor expansion to
        make a small change to the smeared ePSF.  Improves performance
        while this routine is optimized with many small steps.
    ipix_in : ndarray or None, optional
        Boolean array for which indices are actually computed with a
        full chi squared calculation
    return_ancillary : bool, optional
        If True, return a dictionary with ancillary information.
        Default False
    
    Returns
    -------
    chisq_tot : float
        Total chi squared value of a fit of a static scene plus a flux times
        a moving object template defined by the input parameters.  Returned
        if return_flux, return_template, and return_chi2array are all False
    results : dict, optional
        Dictionary with the best flux, its standard error, the templates
        for the moving source, and the per pixel chi squared values.  Only
        returned if return_ancillary is True.
        
    """
    
    _phi, _dist, _x0, _y0 = p

    if psf_saver is not None and psf_saver.chisq_matrix is None:
        psf_saver.chisq_matrix = get_chisq_static(
            diffs, diffs2use, sig, resultants)

    if psf_saver is not None and psf_saver.chisq_matrix is not None:
        ref_chisq = psf_saver.chisq_matrix
    else:
        ref_chisq = None
    
    compute_ipix = ref_chisq is not None
    
    templates_dict = make_templates(epsf,
                                    _phi,
                                    _dist,
                                    _x0,
                                    _y0,
                                    readtimes,
                                    resultants,
                                    oversample,
                                    outshape,
                                    bigpix=compute_ipix,
                                    threshold=1e-4,
                                    psf_saver=psf_saver,
                                    ipix=ipix_in
                                    )

    ipix = templates_dict["ipix"]
    alltemplates = templates_dict["alltemplates"]
    templates_resultants = templates_dict["templates_resultants"]
    templates_resultants_cumsum = templates_dict["templates_resultants_cumsum"]

    Cov = Covar(resultants, templates_resultants_cumsum)
    
    if ref_chisq is not None:
        checkpix = np.zeros(diffs.shape, dtype=bool)
        checkpix[:] = ipix
        res_diffs = diffs[checkpix].reshape((diffs.shape[0], -1))
        res_diffs2use = diffs2use[checkpix].reshape((diffs.shape[0], -1))
    else:
        res_diffs = diffs
        res_diffs2use = diffs2use
        ipix = None

    # Initial guesses for the static scene count rate.

    cguess = np.mean(res_diffs, axis=0)
    cguess[~(cguess > 0)] = 0
    
    # Initial guess of zero for the moving source flux.
    
    fluxguess = np.zeros(res_diffs[0].shape)

    # Two iterations: first without accounting for the contribution of
    # the moving source to the covariance, and then with that accounting.
    
    for i_iter in range(2):
        result = fitramp_movingsource.fit_ramps(
            res_diffs, 
            res_diffs2use,
            templates_resultants/Cov.delta_t[:, None],
            Cov.alpha_phnoise,
            Cov.beta_phnoise,
            Cov.alpha_readnoise,
            Cov.beta_readnoise,
            Cov.alpha_phnoise_path,
            Cov.beta_phnoise_path,
            sig.flatten()[ipix].flatten(),
            cguess,
            fluxguess,
            res_diffs.shape[1],
            res_diffs.shape[0]
        )
        best_flux = np.sum(result["countrate_sec"]/result["uncert_sec"]**2)/np.sum(1/result["uncert_sec"]**2)
        best_flux_err = np.sqrt(1/np.sum(1/result["uncert_sec"]**2))

        # Update the guesses for the static count rate, moving object flux
        
        cguess = (result["A_a"] - best_flux*result["A_ab"])/result["A_aa"]
        cguess[~(cguess > 0)] = 0
        fluxguess = np.ones(res_diffs[0].shape)*best_flux

        # The chi squared is the best chi squared plus the deviations of
        # the pixels from the predicted flux plus the pixels with little
        # contribution from a moving source that were not fitted directly.
        
        chisq_tot = np.sum(result["chisq"]) + np.sum((result["countrate_sec"] - best_flux)**2/result["uncert_sec"]**2)
        if ipix is not None:
            chisq_tot += np.sum(ref_chisq[~ipix])

    if return_ancillary:
        results = {"best_flux" : best_flux,
                   "best_flux_err" : best_flux_err,
                   "alltemplates" : alltemplates,
                   "templates_resultants" : templates_resultants,
                   "templates_resultants_cumsum" : templates_resultants_cumsum
                   }
        return chisq_tot, results
    else:
        return chisq_tot
    

def get_chisq_static(diffs, diffs2use, sig, resultants):

    """
    Compute per-pixel chi squared without a moving source

    Inputs
    ------
    diffs : ndarray
        Scaled differences between consecutive resultants; the scaling is
        the inverse of the difference in mean read time.
        Shape (nresultants - 1, npixels).
    diffs2use : ndarray
        Boolean array of the same shape as diffs indicating which resultant
        differences should be used in the fit
    sig : ndarray
        ndarray of read noise.  Should be of total size npixels.
    resultants : list
        List of values or lists for the indices of reads.  If a list of
        lists, indices for reads that are averaged together to produce
        a resultant.

    Returns
    -------
    chisq : ndarray
        Chi squared value for each pixel without a moving source
    
    """
    
    ndiffs, npix = diffs.shape
    
    # Covariance matrix with zeros for the extra source term
    Cov = Covar(resultants, [np.zeros((len(r), npix)) for r in resultants])
    
    cguess = np.mean(diffs, axis=0)
    cguess[~(cguess > 0)] = 0

    # Zero flux for the extra source
    fluxguess = np.zeros(npix)
    
    for i_iter in range(2):
        result = fitramp_movingsource.fit_ramps(
            diffs, 
            diffs2use,
            np.zeros((ndiffs, npix)),
            Cov.alpha_phnoise,
            Cov.beta_phnoise,
            Cov.alpha_readnoise,
            Cov.beta_readnoise,
            Cov.alpha_phnoise_path,
            Cov.beta_phnoise_path,
            sig.flatten(),
            cguess,
            fluxguess,
            npix,
            ndiffs
        )
        cguess = result["A_a"]/result["A_aa"]
        cguess[~(cguess > 0)] = 0

    return result["chisq"]

    
class Chisq_SavedState:

    """
    Convenience class to avoid repeat computations in the chi2 optimization
    """
    
    def __init__(self):
        self.last_phi = None
        self.last_dist = None

        self.last_smeared_epsf = None
        self.dim_dphi = None
        self.dim_ddist = None

        self.chisq_matrix = None


class MovingTrack:

    """
    Wrapper to generate and fit tracks of moving objects
    """
    
    def __init__(self, readtimes, resultants, epsf, oversample, shape):
        """
        Initialize the MovingTrack object
        
        Inputs
        ------
        readtimes : list
            The indices of all of the individual reads that are binned to
            construct the resultants
        resultants : list
            List of values or lists for the indices of reads.  If a list of
            lists, indices for reads that are averaged together to produce
            a resultant.
        epsf : ndarray
            Two dimensional ndarray represented the effective PSF
        oversample : int
            Factor by which epsf oversamples the output image
        shape : tuple
            Shape of the output array in which to draw/fit the moving source

        Returns
        -------
        None

        Initializes internal attributes to the arguments above.
        """
        self.params = None
        self.readtimes = readtimes
        self.resultants = resultants
        self.epsf = epsf
        self.oversample = oversample
        self.shape = shape

        self.nreads = len(self.readtimes)
        self.nresultants = len(self.resultants)
        
        self.reads_shape = tuple([self.nreads] + list(shape))
        self.resultants_shape = tuple([self.nresultants] + list(shape))
        self.resultants_diff_shape = tuple([self.nresultants - 1] + list(shape))
        self.flatdiffshape = (self.nresultants - 1, np.prod(shape))

        self.alltemplates = None
        self.templates_resultants = None

        self.flux = None
        self.flux_err = None
        self.read_values = None
        self.resultant_values = None
        
    def gen_track(self, params, flux=None, addnoise=False):
        """
        Generate a track of a moving object, both in reads and resultants

        Inputs:
        -------
        params : list
            List of the nonlinear parameters of the moving object:
            phi [angle], dist [distance moved, pix/s],
            x0 [pixel position at t=0], y0 [pixel position at t=0]
        flux : float or None, optional
            Flux in counts/s of the moving object.  If None, fall back on
            self.flux if that is not None.  If both are None, use 1.
        addnoise : bool, optional
            Add photon noise to the moving object's track?  Default False.

        Returns:
        --------
        None

        This routine assigns values to the following attributes:

        self.templates_reads : ndarray
            moving object template, counts in each read difference
        self.templates_resultants : ndarray
            moving object template, counts in each resultant difference
        self.read_values : ndarray
            moving object track, cumulative counts in each read
        self.resultant_values : ndarray
            moving object track, cumulative counts in each resultant
        
        """
        
        phi, dist, x0, y0 = params
        res = make_templates(self.epsf,
                             phi,
                             dist,
                             x0,
                             y0,
                             self.readtimes,
                             self.resultants,
                             self.oversample,
                             self.shape,
                             bigpix=False,
                             ipix=None
                             )
        self.templates_reads = res["alltemplates"].reshape(self.reads_shape)
        self.templates_resultants = res["templates_resultants"].reshape(self.resultants_diff_shape)
        
        diffs = self.templates_reads[1:]*1.
        if flux is not None:
            diffs *= flux
        elif self.flux is not None:
            diffs *= self.flux

        if addnoise:
            diffs = np.random.poisson(diffs)
            
        self.read_values = np.zeros(self.reads_shape)
        self.read_values[1:] = np.cumsum(diffs, axis=0)
        self.resultant_values = bin_to_resultants(self.read_values, self.resultants)
        
    def fit_track(self, params_guess, scaled_diffs, diffs2use, sig_readnoise, method='Nelder-Mead'):
        """
        Fit the track of a moving object to an array of resultant differences

        Inputs:
        -------
        params_guess : list
            Starting guess for the nonlinear parameters of the moving object:
            phi [angle], dist [distance moved, pix/s],
            x0 [pixel position at t=0], y0 [pixel position at t=0]
        scaled_diffs : ndarray
            Scaled differences between consecutive resultants; the scaling is
            the inverse of the difference in mean read time.  Can be either
            2D or 3D; the first dimension is the number of resultants minus 1.
        diffs2use : ndarray
            Boolean array of the same shape as scaled_diffs indicating which
            resultant differences should be used in the fit
        sig_readnoise : ndarray
            ndarray of read noise.  Should be of total size matching
            scaled_diffs[0].
        method : str, optional
            method to be passed to scipy.optimize.minimize.  Default
            'Nelder-Mead'.

        Returns:
        --------
        None

        This routine assigns values to the following attributes:
        self.params : list
            The best-fit nonlinear parameters of the moving object:
            phi [angle], dist [distance moved, pix/s],
            x0 [pixel position at t=0], y0 [pixel position at t=0]
        self.flux : float
            Best-fit flux of the moving object in counts/s.
        self.flux_err : float
            Standard error in self.flux
        self.templates_reads : ndarray
            moving object template, counts/flux in each read difference
        self.templates_resultants : ndarray
            moving object template, counts/flux in each resultant difference
        self.read_values : ndarray
            moving object best-fit track, cumulative counts in each read
        self.resultant_values : ndarray
            moving object best-fit track, cumulative counts in each resultant
        
        """

        if not np.prod(self.flatdiffshape) == np.prod(scaled_diffs.shape):
            raise ValueError("Shape of resultant differences incompatible"
                             " with setup of MovingTrack")
        
        savedstate = Chisq_SavedState()
        
        res = optimize.minimize(full_chisq,
                                params_guess, 
                                args=(scaled_diffs.reshape(self.flatdiffshape),
                                      diffs2use.reshape(self.flatdiffshape),
                                      self.epsf,
                                      sig_readnoise.flatten(),
                                      self.readtimes,
                                      self.resultants,
                                      self.oversample,
                                      self.shape,
                                      savedstate),
                                method=method)

        self.params = res.x
        _, ancillary = full_chisq(self.params,
                                  scaled_diffs.reshape(self.flatdiffshape),
                                  diffs2use.reshape(self.flatdiffshape),
                                  self.epsf,
                                  sig_readnoise.flatten(),
                                  self.readtimes,
                                  self.resultants, 
                                  self.oversample,
                                  self.shape,
                                  return_ancillary=True
                                  )
        self.flux = ancillary["best_flux"]
        self.flux_err = ancillary["best_flux_err"]

        self.gen_track(self.params)
