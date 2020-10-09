import numpy as np

from . import config as cfg

def bnu_wav_micron(wav_um,temp):
    """Return a Planck function, avoiding overflows.
    
    Parameters
    ----------
    wave_um : ndarray of float
        Wavelengths at which to compute flux.
    temp : float
        Temperature of blackbody.
    """
    k1 = 3.9728949e19
    k2 = 14387.69
    fact1 = k1/(wav_um**3)
    fact2 = k2/(wav_um*temp)
    if isinstance(wav_um,np.ndarray):
        ofl = fact2 < 709
        bnu = np.zeros(len(fact2)) + cfg.tiny
        if np.any(ofl) == False:
            return bnu
        else:
            bnu[ofl] = fact1[ofl]/(np.exp(fact2[ofl])-1.0)
            return bnu
    elif isinstance(temp,np.ndarray):
        ofl = fact2 < 709
        bnu = np.zeros(len(fact2)) + cfg.tiny
        if np.any(ofl) == False:
            return bnu
        else:
            bnu[ofl] = fact1/(np.exp(fact2[ofl])-1.0)
            return bnu
    else:
        if fact2 > 709:
            return cfg.tiny
        else:
            return fact1/(np.exp(fact2)-1.0)


def rotation_profile(wave, center, v, depth, w_true=1):
    '''Return a stellar rotation line profile.

    See 1933MNRAS..93..478C.

    Parameters
    ----------
    wave : ndarray of float
        Wavelengths at which to compute the spectrum.
    v : float
        V sin i of the star in km/s.
    w_true : float
        True width of line in km/s (assumed Gaussian).
    '''

    c = 299792.458 # km/s
    beta = v/c
    xi = (wave - center)/center
    nxi = len(xi)
    
   
    def I(xi):
        '''true line shape'''
        return 1-depth*np.exp(-(xi/(2*w_true/c))**2)

    t = np.linspace(-1,1,1000)
    spec = np.zeros(nxi)
    for i,x in enumerate(xi):
        spec[i] = 2/np.pi*np.trapz(I(x+beta*t) * np.sqrt(1-t**2), x=t)

    return spec
