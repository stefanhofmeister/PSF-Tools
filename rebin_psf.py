#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:06:28 2023

@author: sjh
"""

import numpy as np

def rebin_psf(psf, dimension):
    """
    Rebin the PSF to smaller dimensions

    If one wants to deconvolve a rebinned image with the PSF, the PSF has to be rebinned accordingly. However, the definition of the center of the PSF is not in the center of the 2d PSF array, but shifted by definition by half a pixel. This function accounts for this shift for rebinning the PSF.

    Parameters
    ----------
    psf : `~numpy.ndarray`
        The point spread function. 
    dimensions : [xdim, ydim]
        The new size of the PSF. 
    
    Returns
    -------
    np.ndarray
        The the rebinned PSF

   """
   
    psf = np.copy(psf)
    psf_rebinned = np.zeros(dimension)
    
    shape_fac = (psf.shape / np.array(dimension)).astype(np.int)
    dx_l = shape_fac[0]//2
    dy_l = shape_fac[1]//2
    
    for dx in range(dx_l):
        psf = np.append(np.array([psf[:, -dx-1]]).transpose(), psf, axis = 1)  #add an extra column, so that the summation is eased. Adding the values of the last column is technically not absolute correct, but there is no exact solution to this process. Using the last column conserves the total PSF weight.
    for dy in range(dy_l):
        psf = np.append([psf[-dy-1, :]], psf, axis = 0)
        
    for dx in range(-dx_l+1, dx_l):
      for dy in range(-dy_l+1, dy_l):
          psf_rebinned += psf[dx_l+dx:-(dx_l-dx):shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
    
    for dy in range(-dy_l+1, dy_l):
          psf_rebinned += 0.5*psf[0:-2*dx_l:shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
          psf_rebinned += 0.5*psf[2*dx_l::shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
    for dx in range(-dx_l+1, dx_l):
          psf_rebinned += 0.5*psf[dx_l+dx:-(dx_l-dx):shape_fac[0], 0:-2*dy_l:shape_fac[1]]
          psf_rebinned += 0.5*psf[dx_l+dx:-(dx_l-dx):shape_fac[0], 2*dy_l::shape_fac[1]]
     
    psf_rebinned += 0.25*psf[0:-2*dx_l:shape_fac[0], 0:-2*dy_l:shape_fac[1]]
    psf_rebinned += 0.25*psf[2*dx_l::shape_fac[0], 0:-2*dy_l:shape_fac[1]]
    psf_rebinned += 0.25*psf[0:-2*dx_l:shape_fac[0], 2*dy_l::shape_fac[1]]
    psf_rebinned += 0.25*psf[2*dx_l::shape_fac[0], 2*dy_l::shape_fac[1]]
    
    return psf_rebinned