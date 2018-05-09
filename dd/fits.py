'''Functions for dealing with FITS related stuff.'''

import os
import numpy as np
from astropy.io import fits

def image2fits(image, arcsec_pix, fits_file, bunit='Jy/pixel',
               total_flux=None, peak_flux=None,
               ra_deg=None, dec_deg=None, overwrite=False):
    '''Write a simple FITS from from a 2d array.
    
    Parameters
    ----------
    array : np.ndarray
        The image.
    arcsec_pix : float
        Pixel scale of the image.
    fits_file : str
        Name of FITS file to write to.
    bunit : str, optional
        Units of the output file.
    total_flux : float, optional
        Total flux in bunit of the output file.
    peak_flux : float, optional
        Peak flux in bunit of the ouput file.
    ra_deg : float
        Right ascension of the image center in degrees.
    dec_deg : float
        Declination of the image center in degrees.
    overwrite : bool, optional
         Overwrite FITS file.
    '''

    if os.path.exists(fits_file) and not overwrite:
        raise ValueError('{} exists, but overwrite set to {}'.format(fits_file,overwrite))

    # make a simple FITS file
    hdu = fits.PrimaryHDU(image)
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    hdu.header['CDELT1'] = -arcsec_pix / 3600
    hdu.header['CDELT2'] = arcsec_pix / 3600
    hdu.header['CRPIX1'] = image.shape[1]/2.
    hdu.header['CRPIX2'] = image.shape[0]/2.
    if ra_deg is not None and dec_deg is not None:
        hdu.header['CRVAL1'] = ra_deg
        hdu.header['CRVAL2'] = dec_deg
    hdu.header['CROTA1'] = 0.0
    hdu.header['CROTA2'] = 0.0
    hdu.header['BUNIT'] = 'Jy/pixel'
    hdu.writeto(fits_file, overwrite=overwrite)


def phantomasciigrid2fits(grid, x_size, fits_file, bunit='Jy/pixel',
                          total_flux=None, peak_flux=None,
                          image_axis=0, skiprows=19,
                          ra_deg=None, dec_deg=None, overwrite=False):
    '''Convert a phantom ascii grid file to a flattened FITS file.
    
    The grid files are created using ssplash to gridascii dump_XXXX.
    Reading files in is slow, sadly. To figure out the pixel scale the
    output as ssplash is run needs to be checked, the units are likely
    to be au.
    
    Most of the keywords are passed to image2fits.
    
    Parameters
    ----------
    grid : str
        Name of the grid file.
    x_size : float
        Size of the x dimension in au.
    fits_file : str
        Name of FITS file to write to.
    bunit : str, optional
        Units of the output file.
    total_flux : float, optional
        Total flux in bunit of the output file.
    peak_flux : float, optional
        Peak flux in bunit of the ouput file.
    image_axis : int, optional
        Axis to create image along (0=z, 1=y, 2=z)
    skiprows : int, optional
        Number of rows before data starts.
    overwrite : bool, optional
         Overwrite FITS file.
    '''

    if os.path.exists(fits_file) and not overwrite:
        raise ValueError('{} exists, but overwrite set to {}'.format(fits_file,overwrite))

    # read the file and make a cube
    nx, ny, nz = np.genfromtxt(grid, max_rows=1, dtype=int)
    d = np.loadtxt(grid, skiprows=skiprows)
    d = d.reshape(nz, ny, nx)

    # squash to 2d
    im = np.sum(d, axis=image_axis)

    # normalise
    if total_flux is not None:
        im *= total_flux / np.sum(im)
    if peak_flux is not None:
        im *= peak_flux / np.max(im)

    arcsec_pix = x_size / float(im.shape[1])

    image2fits(im, arcsec_pix, fits_file, bunit='Jy/pixel',
               total_flux=None, peak_flux=None,
               ra_deg=None, dec_deg=None, overwrite=False)

    return d
