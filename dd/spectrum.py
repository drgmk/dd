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


def col2temp(wav_um, flux):
    '''Convert two fluxes at different wavelengths to a blackbody temp.
    
    Parameters
    ----------
    wav_um: length 2 array or tuple
        Wavelengths of two fluxes
    flux: length 2 array or tuple
        Fluxes in same units
    '''
    
    if wav_um[0] > wav_um[1]:
        wav = [wav_um[1],wav_um[0]]
        flx = [flux[1], flux[0]]
    
    for t in np.arange(1,1000):
    
        f1 = bnu_wav_micron(wav[0], t)
        f2 = bnu_wav_micron(wav[1], t)
        
        if f1/f2 < flx[0]/flx[1]:
            return t


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


def rprof(p, rv, ccf, model=False):
    '''Function to return chi^2 for a CCF model, with rotation.
    
    Uncertainties/variation in CCF not taken into account.
    
    Parameters
    ----------
    p: list
        Parameters (km/s); rv center, vsini, depth (assuming visni=0), true line width
    rv: array
        Radial velocities for CCF
    ccf: array
        CCF
    model: bool
        Return model, not chi^2
    '''
    w = (rv / 299792.458 + 1)
    l = np.polyval([p[5],p[4]], w-1)
    m = dd.spectrum.rotation_profile(w, p[0], p[1], p[2], p[3]) + l
    if model:
        return m
    chi2 = np.sum( (ccf - m)**2 )/0.05**2
    return chi2


def ln_rprof(p, rv, ccf):
    '''Log likelihood for rprof function.'''
    if np.any(p[1:4] < 0):
        return -np.inf
    if np.abs(p[0]) > 0.1 or p[1] > 200 or p[3] > 10:
        return -np.inf
    return -0.5 * rprof(p, rv, ccf)


def ccf_rv(file, rvc=0, rvw=150, nrv=300):
    '''Compute a CCF using raccoon and estimate RV.
    
    Parameters
    ----------
    file : str
        Path to FITS file
    rvc : float
        Center of RV range
    rvw : float
        Half of CCF range
    nrv : int
        Number of points in CCF
    '''
    
    # open the file, assume HARPS like
    spec = fits.getdata(file)
    wave = spec['WAVE'][0]
    flux = spec['FLUX'][0]
    
    # do CCF, cut line list to spectrum
    rv = np.linspace(rvc-rvw, rvc+rvw, num=nrv)
    wsh1 = (wave[0] -(wave[1] -wave[0] )/2)*(1-(rvc-rvw)/2.99792458e5)
    wsh2 = (wave[-1]+(wave[-1]-wave[-2])/2)*(1-(rvc+rvw)/2.99792458e5)
    ok = (wm > wsh1) & (wm < wsh2)
    ccf, _ = raccoon.ccf.computeccf(wave, flux, np.ones_like(wave), wm[ok], fm[ok], rv)
    ccfsave = ccf.copy()
    
    # estimate RV fit starting parameters
    ccf = ccf - np.min(ccf)
    ccf = ccf/np.median(ccf)
    w = (rv / 299792.458 + 1)
    slope = (np.mean(ccf[-10:])-np.mean(ccf[:10])) / \
            (np.mean(w[-10:])  -np.mean(w[:10]))
    p0 = [w[np.argmin(ccf)], 10., 1-np.min(ccf), 1.,
          (np.min(w)-1)*slope, slope]
    p0 = np.array(p0)
    
    # fitting
#     x = scipy.optimize.minimize(rprof, p0, args=(rv,ccf))
#     par = x['x']
    
    nwalkers, ndim, nstep, burn = 32, 6, 500, 400
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_rprof, args=(rv,ccf))
    mult = np.array([1e-5, 0.1, 0.1, 0.1, 0.1, 0.1])
    pos = [p0 + p0*mult*np.random.randn(ndim) for i in range(nwalkers)]
    pos, prob, state = sampler.run_mcmc(pos, nstep, progress=True)
    par = np.median(sampler.chain[:,-1,:].reshape((-1,ndim)),axis=0)
    rvcen = (par[0]-1)*3e5
    resid = ccf - rprof(par, rv, ccf, model=True)
    
    # plot emcee sampling
    fig,ax = plt.subplots(ndim+1,2,figsize=(9.5,5),sharex='col',sharey=False)
    for j in range(nwalkers):
        ax[-1,0].plot(sampler.lnprobability[j,:burn])
        for i in range(ndim):
            ax[i,0].plot(sampler.chain[j,:burn,i])
    for j in range(nwalkers):
        ax[-1,1].plot(sampler.lnprobability[j,burn:])
        for i in range(ndim):
            ax[i,1].plot(sampler.chain[j,burn:,i])
    ax[-1,0].set_xlabel('burn in')
    ax[-1,1].set_xlabel('sampling')
    fig.savefig(f'{file}-mcmc.png')
    plt.close(fig)
    # plot final fit
    fig, ax = plt.subplots(figsize=(9.5,5))
    ax.plot(rv, ccf)
    ax.plot(rv, rprof(par, rv, ccf, model=True))
    ax.plot(rv, rprof(p0, rv, ccf, model=True), alpha=0.5)
    fig.savefig(f'{file}-fit.png')
    fig.tight_layout()
    plt.close(fig)

    np.savez(f'{file}-ccf.npz', rv, ccfsave, sampler.chain, resid)

    return rv, ccf, rvcen, par, p0, resid
