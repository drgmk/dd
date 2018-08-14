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
