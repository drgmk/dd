import numpy as np

def prtau(mstar, r0, tau0, beta=0.5, k=1.0, nr=1000):
    '''Get the optical depth profile for the Wyatt 2005 P-R drag model.
    
    Parameters
    ----------
    mstar : float
        Stellar mass.
    r0 : float
        Location of inner edge of source belt.
    tau0 : float
        Face-on optical depth of source belt.
    beta = float, optional
        Radiation/gravity force ratio for dust particles.
    k : float, optional
        Additional collisional depletion factor (Kennedy & Piette).
    nr : int, optional
        Number of radii to compute profile for.
    '''

    # eta parameter
    eta0 = tau0 * 5000.0 * np.sqrt(r0) / (np.sqrt(mstar) * beta * k)

    # set up radii and figure optical depth
    rdash = np.linspace(0, nr, nr)/float(nr)
    rdash_cen = (rdash[:-1] + rdash[1:])/2
    rdcsq = 1.0 - np.sqrt(rdash_cen)
    r_cen = rdash_cen * r0
    taur = tau0 / (1 + 4 * eta0 * rdcsq)

    return r_cen, taur
