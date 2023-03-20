import sunpy.map
import numpy as np
import scipy as scp
import astropy.units as u
import skimage.morphology as morph
import skimage.filters as filter
import skimage.transform as transform
import skimage.feature as feature
import copy
import os
from functools import partial
import os.path
from numba import jit, njit
from  multiprocessing import Pool
import sys


def rebin_allfiles(folder_top, files, resolution, workers = 1, plot = 0):
    """
    Sets up the environment for file rebinning associated with the revised PSF study, and rebins the files.
    """
    if isinstance(files, list) == False: files = [files]
    files = [folder_top+'/original/' + os.path.splitext(os.path.basename(f))[0] + '.fits' for f in files]
    folder_out = folder_top + '/rebinned/'
    rebin_file(files, folder_out, resolution, plot = plot, workers = workers)




def rebin_file(files, outdir, resolution, conserve_flux = True, plot = False, dpi = 300, workers = 1):
    """
    Rebin fits files to a lower resolution, and save the rebinned files in the outfolder.

    Parameters
    ----------
    files : array of filenames
        Files to rebin.
    outdir : path
        Directory to put the rebinned files in. If only one resolution was specified, the files are directly put into this directory. If multiple resolutions are specified, subdirectories are automatically created.
    resolution : int / array of ints
        Resolutions to that the files shall be rebinned.
    conserve_flux : True/False
        Specifies if the total flux shall be conserved. If set to false, the flux density will be conserved. The default is True.
    plot : True/False, optional
        If set to true, also create overview plots. The default is False.
    dpi : int, optional
        Resolution of the overview plot in dpi. The default is 300.
    workers : int, optional
        Number of workers, if the rebining is wanted to be done using multiple cpu cores. The default is 1.

    Returns
    -------
    None.

    """
    #create the target directories
    if not isinstance(resolution, list):
        os.makedirs(outdir, exist_ok = True)    
    else:
        for res in resolution: os.makedirs(outdir + '/' + str(res), exist_ok = True)    
    if isinstance(files, list) == False: files = [files]
    #distribute the files to the workers
    rebin_file_chunk_partial = partial(rebin_file_chunk, outdir_top = outdir, resolution=resolution, conserve_flux = conserve_flux, plot = plot, dpi = dpi) 
    with Pool(workers) as pool:
          pool.map(rebin_file_chunk_partial, files)
    
def rebin_file_chunk(file, outdir_top, resolution, conserve_flux, plot, dpi):     
    """
    Helper function for rebin_file. In fact, all the work is done in here.
    """
    map_in = sunpy.map.Map(file)
    if not isinstance(resolution, list): resols = [resolution]
    else: resols = resolution
    for res in resols:      
        map_rbin = rebin_map(map_in, (res, res), conserve_flux = conserve_flux)  
           
        if not isinstance(resolution, list): outdir = outdir_top
        else: outdir = outdir_top + '/' + str(res)
        
        file_fits = outdir + '/' + os.path.basename(file)
        map_rbin.save(file_fits)
        
        if plot:
            # print(map_rbin)
            #sys.exit()

            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = plt.figure()
            ax = plt.subplot(111, projection=map_rbin.wcs)
            
            file_img = outdir + '/' + os.path.splitext(os.path.basename(file))[0] + '.jpg'        
            map_rbin.plot()
            plt.savefig(file_img, dpi = dpi)
            plt.close()




def rebin(array, dimension, conserve_flux = True): 
    """
    Rebin an array to larger/smaller resolutions. Meant for 2d images, but can be used for arrays of arbitrary dimensions. By default, the total flux is conserved

    Parameters
    ----------
    array : nd array of ints/floats
        Input array to be rebinned
    dimension : array of ints
        New dimensions of the array. Larger and smaller dimensions than the original dimensions are allowed. But, the p/downscaling factor, i.e.,  array.shape/dimensions or dimensions/array.shape, respectively, has to be an integer.
    conserve_flux : True/False, optional
        Specifies if the total flux shall be conserved. If set to false, the flux density will be conserved. The default is True.

    Returns
    -------
    array : nd array
        Rebinned array.

    """
    array = np.array(array)
    shape_in = np.array(array).shape
    dimension = np.array(dimension).astype(int) 
    
    # print(array.shape, shape_in, dimension)
    
    #first, rebin the array to lower resolutions where required
    #create the target shape array
    shape = list()
    for s, r in zip(shape_in, dimension):
        if s/r > 1: shape.extend([r, int(s/r)])
        else: shape.extend([s, 1])
    #now, reshape the array, and sum over the uneven dimensions
    array = array.reshape(shape)
    # print(dimension, len(dimension), array.shape, shape, shape_in, dimension)

    for i in reversed(range(len(dimension))):
       array = array.sum(2*i+1)
    #next, rebin the array to higher resolutions where required
    for i, (s, r) in enumerate(zip(shape_in, dimension)):
        if s/r < 1:
            array = array.repeat(r/s, axis = i) / (r/s)
    #if the pixel values should be preserved instead of the total flux, rescale the array
    if conserve_flux == False:
        array /= np.prod(np.divide(shape_in, dimension))
    return array
      




def rebin_map(map_in, dimension, conserve_flux=True):
    """
    Rebin a map to the specified dimensions. At the moment, only downscaling is allowed.

    Parameters
    ----------
    map : map of an astropy/sunpy fits file
        Input map.
    dimension : tuple of integers
        New dimensions of the map.
    conserve_flux : True/False, optional
        Specifies if the total flux shall be conserved. If set to false, the flux density will be conserved. The default is True.

    Returns
    -------
    map_rebinned : map
        Rebinned map.

    """
    
    map_rebinned = map_in.resample(dimension*u.pixel) 
    map_rebinned.data[:, :] = rebin(map_in.data, dimension, conserve_flux = conserve_flux)
    
    # res_map = map.data.shape
    # shape_factor = np.array(res_map) / dimension
    # if conserve_flux == True:
    #     map_rebinned = copy.deepcopy(map).superpixel(shape_factor*u.pixel, func = np.sum) 
    # if conserve_flux == False:
    #     map_rebinned = copy.deepcopy(map).superpixel(shape_factor*u.pixel, func = np.mean)
    return map_rebinned






@jit(parallel = False, nopython = True)
def despike_map(sunpymap, kernel=3,  sigma = 5, itmax = 20, verbose = True, threshold = 0):
    """
    Despike sunpymap.

    Parameters
    ----------
    sunpymap : sunpymap of an astropy/sunpy fits file
        Input sunpymap.
    kernel : int, optional
        Size of the kernel, in which for spikes is looked. The default is 3.
    sigma : float, optional
        Pixel values exceeding sigma standard deviations are presumed to be spikes, and replaced by the mean of the kernel. The default is 5.
    itmax : int, optional
        Maximum number of iterations to perform for despiking. Despiking stops automatically when no more spikes are found. The default is 20.
    verbose : True/False, optional
        Prints the number of spikes found in each iteration. The default is False.
    threshold : float, optional
        Pixel values which absolute value is smaller than threshold are never despiked. The default is 0.

    Returns
    -------
    sunpymap : sunpymap
        The despiked sunpymap.

    """
    
    # data = sunpymap.data
    data = sunpymap
    kernel_rad = int((kernel -1)/2.)

    data_out = np.copy(data)
    for it in range(itmax):
        n_repl = 0.
        for i in range(len(data[:, 0])):
            for k in range(len(data[0, :])):
                if (i < kernel_rad ) or (i > len(data[:, 0]) - kernel_rad -1): continue
                if (k < kernel_rad ) or (k > len(data[0, :]) - kernel_rad -1): continue
                neighborhood = data[i - kernel_rad : i + kernel_rad+1, k - kernel_rad : k + kernel_rad+1].flatten()
                median = np.median(neighborhood)
                mean_px = (np.sum(neighborhood) - data[i, k]) / (len(neighborhood) -1.)
                neighborhood = np.abs(neighborhood - median)
                sigma_px = np.std(neighborhood)
                if sigma_px < 1: sigma_px = 1
                sigma_px = np.abs((data[i, k] - median) / sigma_px)
                if (sigma_px > sigma) & (np.abs(data[i, k] > threshold) ) : 
                    if data_out[i,k] == mean_px: continue
                    data_out[i, k] = mean_px
                    n_repl += 1
        data = np.copy(data_out)
        # if verbose:
        #     print('In iteration {:}, {} pixels have been replaced.'.format(it, n_repl))            
        if n_repl == 0.: break
    # sunpymap.data[: , :] = data
    return data# sunpymap

