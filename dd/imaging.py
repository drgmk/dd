import numpy as np

def gauss_2d(x_off, y_off, peak, sig_maj, sig_min, pa, size, norm=False):
    '''Return a 2d Gaussian.
    
    Parameters
    ----------
    x_off : float
        X pixel offset from center.
    y_off : float
        Y pixel offset from center.
    peak : float
        Peak value.
    sig_maj : float
        Major axis sigma.
    sig_min : float
        Minor axis sigma.
    pa : float
        PA E of N in degrees.
    size : tuple
        Size of image (y,x).
    norm : bool, optional
        Normalise total to 1.
    '''
    ny, nx = size
    yc, xc = (ny-1)/2, (nx-1)/2
    x, y = np.meshgrid(np.arange(nx)-xc-x_off, np.arange(ny)-yc-y_off)
    xp = x * np.cos(np.deg2rad(pa)) - y * np.sin(np.deg2rad(pa))
    yp = x * np.sin(np.deg2rad(pa)) + y * np.cos(np.deg2rad(pa))
    g = peak * np.exp( -(((xp)/sig_maj)**2+
                         ((yp)/sig_min)**2)/2.)
    if norm:
        g /= np.sum(g)

    return g
