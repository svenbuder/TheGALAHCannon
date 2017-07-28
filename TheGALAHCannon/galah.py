""" Functions for reading in GALAH spectra and training labels """

from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
from TheGALAHCannon import *

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def load_labels(cannon_version,labels,labels_ref):
    """ Extracts reference labels from a file
        
        Parameters
        ----------
        cannon_version: str
            Name of the Cannon version containing the FITS file with reference labels
        
        labels: array
            The labels to choose retrieve
        
        labels_ref: array
        
        Returns
        -------
        tr_ID: ndarray
            The identifications for each reference object
        
        tr_labels: array
            Reference label values for all reference objects
        """
    print("Loading reference labels from file %s" %cannon_version+'/'+cannon_version+'_trainingset.fits')
    door = pyfits.open(cannon_version+'/'+cannon_version+'_trainingset.fits')
    data = door[1].data
    door.close()
    ids = data['sobject_id']
    inds = ids.argsort()
    ids = ids[inds]
    teff = data['TEFF']
    logg = data['LOGG']
    feh  = data['FEH']
    print("OBS: Here we have to get a general choice from the labels, right now it is TEFF,LOGG,FEH")
    tr_labels = np.vstack((teff,logg,feh)).T
    return ids, tr_labels

def load_spectra(tr_ID,wl_grid):
    """ Reads wavelength, flux, and flux uncertainty data from GALAH spectrum fits and interpolates them onto given wavelength (wl_grid)

    Parameters
    ----------
    tr_ID: ndarray
        The identifications for each reference object
    
    wl_grid: array
        Wavelength grid onto which the spectra are interpolated at rest
        
    Returns
    -------
    fluxes: ndarray
        Flux data values

    ivars: ndarray
        Inverse variance values corresponding to flux values
    """
    nstars = len(tr_ID)
    npixels = len(wl_grid)

    fluxes = 1.0*np.ones((nstars, npixels), dtype=float)
    ivars  = 0.000001*np.ones((nstars, npixels), dtype=float)

    for jj, each_ID in enumerate(tr_ID):

        for each_ccd in [1,2,3,4]:
    
            # First get information from FITS file
            fits_ccd = pyfits.open('SPECTRA/dr5.2/'+str(each_ID)[0:6]+'/standard/com/'+str(each_ID)+str(each_ccd)+'.fits')
            flux = np.array(fits_ccd[4].data)
            flux_err = np.array(fits_ccd[1].data)
            if each_ccd == 4:
                flux_err = flux_err[len(flux) - len(flux_err):]
            start_wl = file_in[4].header['CRVAL1']
            diff_wl = file_in[4].header['CDELT1']
            ccd_pixels = file_in[4].header['NAXIS']
            fits_ccd.close()

            wl_ccd     = diff_wl * (ccd_pixels) + start_wl

            # Then interpolate onto wl_grid
            
            fluxes[jj,:] += np.interp(wl_grid,wl_ccd,flux)
            flux_err = np.interp(wl_grid,wl_ccd,flux_err)
            ivars[jj,:] += 1. / flux_err**2
    print("Spectra loaded")
    return fluxes, ivars

if  __name__ =='__main__':
    make_galah_label_file()
