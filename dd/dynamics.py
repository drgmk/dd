from functools import lru_cache
import scipy.interpolate
import numpy as np
from astropy import constants as const


def el2xv(a, e, i_, peri_, node_, f_, mstar=None, degrees=True,
          au=True, return_2d=False):
    '''Convert orbital elements to positions and velocities.
    
    Output length is in same units as input, in au by default when the
    stellar mass is given (in Solar masses) and velocities are computed
    (returned in in m/s).
    
    Parameters
    ----------
    a : ndarray
        Semi-major axis.
    e : ndarray
        Eccentricity.
    i : ndarray
        Inclination.
    peri : ndarray
        Argument of pericenter, in degrees by default.
    node : ndarray
        Longitude of ascending node, in degrees by default.
    f : ndarray
        True anomaly, in degrees by default.
    mstar : float, optional
        Stellar mass, in Solar masses.
    degrees : bool, optional
        Indicate the angles are given in degrees.
    au : bool, optional
        Indicate that a is given in au.
    return_2d : bool, optional
        Return x,y in frame of orbit (pericenter along x).
    '''

    if degrees:
        i = np.deg2rad(i_)
        peri = np.deg2rad(peri_)
        node = np.deg2rad(node_)
        f = np.deg2rad(f_)

    # x and y positions in orbital plane (notes are M&D equation numbers)
    p = a * ( 1. - e**2 )           # 2.20
    r = p / ( 1. + e * np.cos(f) )  # 2.20
    x2d = r * np.cos(f)             # 2.21
    y2d = r * np.sin(f)             # 2.21
    if return_2d:
        return x2d, y2d

    # Equation 2.122
    cn = np.cos(node)
    sn = np.sin(node)
    cp = np.cos(peri)
    sp = np.sin(peri)
    ci = np.cos(i)
    si = np.sin(i)

    # this is the rotation
    x = x2d * ( cn*cp - sn*sp*ci ) + y2d * ( -cn*sp - sn*cp*ci )
    y = x2d * ( sn*cp + cn*sp*ci ) + y2d * ( -sn*sp + cn*cp*ci )
    z = x2d * ( sp*si )            + y2d * (  cp*si )

    # x and y velocities in orbital plane, if needed
    if mstar is None:
        return x,y,z
    else:
        mu = mstar * const.M_sun.si.value * const.G.si.value
            
    vp = np.sqrt(mu/p)
    if au:
        vp /= np.sqrt(const.au.si.value)

    vx2d = -np.sin(f) * vp             # 2.36
    vy2d = (e + np.cos(f)) * vp        # 2.36

    # this is the rotation
    vx = vx2d * ( cn*cp - sn*sp*ci ) + vy2d * ( -cn*sp - sn*cp*ci )
    vy = vx2d * ( sn*cp + cn*sp*ci ) + vy2d * ( -sn*sp + cn*cp*ci )
    vz = vx2d * ( sp*si )            + vy2d * (  cp*si )
  
    return x, y, z, vx, vy, vz


def xv2el(mstar, x_, y_, z_, vx_, vy_, vz_, au=True, return_au=True,
          return_degrees=True, v_au_day=False):
    '''Convert positions and velocities to orbital elements.
    
    Parameters
    ----------
    mstar : float
        Mass of star in Solar masses.
    x, y, z : ndarray
        Cartesian coordinates, in au by default.
    vx, vy, vz : ndarray
        Certesian velocities, in m/s by default.
    au : bool, optional
        Indicate that semi-major axis is given in au, rather than m.
    v_au_day : bool, optional
        Indicate the velocities are given in au/day, rather than m/s.
    return_au : bool, optional
        Return semi-major axis in au, rather than m.
    degrees : bool, optional
        Return angles in degrees, rather than radians.
    '''

    mu = mstar * const.M_sun.si.value * const.G.si.value

    if type(x_) in [int,float]:
        szx = 1
    else:
        szx = len(x_)

    if au:
        x = x_ * const.au.si.value
        y = y_ * const.au.si.value
        z = z_ * const.au.si.value
    else:
        x, y, z = x_, y_, z_

    if v_au_day:
        vx = vx_ * const.au.si.value / (60*60*24)
        vy = vy_ * const.au.si.value / (60*60*24)
        vz = vz_ * const.au.si.value / (60*60*24)
    else:
        vx, vy, vz = vx_, vy_, vz_

    # basic quantities (notes are M&D equation numbers)
    r = np.sqrt( x**2 + y**2 + z**2 )   # 2.126
    v2 = vx**2 + vy**2 + vz**2          # 2.127
    r_dot_rdot = x*vx + y*vy + z*vz     # 2.128
    hvec = np.zeros((3, szx))
    hvec[0,:] = y*vz - z*vy
    hvec[1,:] = z*vx - x*vz
    hvec[2,:] = x*vy - y*vx               # 2.129
    habs2 = hvec[0,:]**2 + hvec[1,:]**2 + hvec[2,:]**2
    habs = np.sqrt( habs2 )             # length of hvec
    rdot = np.sqrt(v2 - (habs/r)**2)    # 2.130
  
    # sign of Rdot is taken as sign of RdotRdot
    rdot[r_dot_rdot < 0] *= -1

    # now get elements
    a = 1.0/( 2./r - v2/mu )         # 2.134
    e = np.sqrt(1.0 - habs2/(mu*a))  # 2.135
    i = np.arccos(hvec[2,:]/habs)    # 2.136
    sini = np.sin(i)
    cosi = np.cos(i)

    # the Hz sign stuff in M&D only works if I is between -90 and 90.
    # Ignore it since habs*sini can also suffer truncation error for
    # low inclinations

    # this is simpler, node is in next quadrant acw from hvec in x/y plane
    node = np.arctan2(hvec[0,:], -hvec[1,:])
    node[node < 0] += 2 * np.pi
    cosn = np.cos(node)
    sinn = np.sin(node)

    # set some things if no inclination
    node[i==0] = 0.
    sini[i==0] = 0.
    cosi[i==0] = 1.
    cosn[i==0] = 1.
    sinn[i==0] = 0.

    # true anomaly like slalib (pv2el)
    s = habs * r_dot_rdot
    c = habs2 - r * mu
    f = np.arctan2(s, c)
    f[f < 0] += 2 * np.pi

    # argument of pericenter like slalib (pv2el)
    u = np.arctan2((-x*sinn+y*cosn)*cosi+z*sini, x*cosn+y*sinn)
    peri = (u - f) % (2 * np.pi)
    peri[peri < 0] += 2 * np.pi

    if return_degrees:
        i = np.rad2deg(i)
        node = np.rad2deg(node)
        peri = np.rad2deg(peri)
        f = np.rad2deg(f)

    # convert a back to AU if needed
    if return_au:
        a /= const.au.si.value

    return a, e, i, peri, node, f


def orbit_rv(nu, q_rstar, t_cen, t, e=1.0, m_star=1.7, r_star=1.5, testing=False):
    '''RVs during transit, parabolic or elliptical.
        
    Parameters
    ----------
    nu : float
    Pericenter angle from center of transit.
    q_rstar : float
    Pericenter in stellar radii.
    t_cen : float
    Time at transit mid-point in hours (where f = -nu).
    t : ndarray
    Times at which to calculate RV.
    e : float
    Orbit eccentricity, if =1 will use parabolic functions.
    '''
    
    # some constants
    mu = const.G.value * m_star * u.M_sun.to('kg')
    q = q_rstar * r_star * u.R_sun.to('m')
    if e == 1:
        h = np.sqrt(2*mu*q)
    else:
        a = q / (1-e)
        h = np.sqrt(mu*a*(1-e**2))
    
    # time to pericenter from transit center, and true anomaly at times t
    if e == 1:
        t_to_cen = t_at_f_para(mu, q, -nu)
        f = f_at_t_para(mu, q, t_to_cen + (t - t_cen)*3600 )
    else:
        t_to_cen = t_at_f_ellip(mu, q, e, -nu)
        f = f_at_t_ellip(mu, q, e, t_to_cen + (t - t_cen)*3600 )
    
    # is planet in front of star?
    if e == 1:
        r = 2 * q / (1+np.cos(f))
    else:
        r = a*(1-e**2)/(1+e*np.cos(f))

    x = r * np.cos(f)
    y = r * np.sin(f)
    bigx =  x * np.sin(nu) + y * np.cos(nu)
    bigy = -x * np.cos(nu) + y * np.sin(nu)
    in_transit = ( np.abs(bigx) < ( r_star * u.R_sun.to('m') ) ) & \
        ( bigy < 0 )

    # compute RVs
    rv = mu / h * (np.sin(f)*np.cos(nu) + np.sin(nu)*(e+np.cos(f)))
        
    if testing:
        return in_transit, rv, t_to_cen, f, r/(r_star * u.R_sun.to('m')), x, bigy
    else:
        return in_transit, rv


def sky_orbit(a, e, i, peri, node_north, epoch=None, primary=False,
                    anomaly=None, t=None, period=None):
    '''Return an orbit on the sky.
    
    x, y are W, N, and z coordinate is toward us.

    Parameters
    ----------
    a : float
        Semi-major axis.
    e : float
        Eccentricity.
    i : float
        Inclination.
    peri : float
        Argument of pericenter, in degrees.
    node_north : float
        Longitude of ascending node, East of North, in degrees.
    epoch : ndarray
        List of times to return positions.
    primary : bool, optional
        Return primary position relative to secondary, not vice versa.
    anomaly : float, optional
        True anomaly, must be given if epoch, T and period are not.
    t : float
        Time of pericenter passage in years, must be given if anomaly is not.
    period : float
        Orbital period in years, must be given if anomaly is not.
    '''

    if anomaly is not None:
        f = anomaly
    else:
        dyr = (epoch - t) % period      # time since last pericenter passage
        m = dyr / period * np.pi*2      # mean anomaly (radians)
        f = np.rad2deg( convmf(m, e) )  # true anomaly (degrees)

    if primary:
        peri = peri + 180

    # convert to cartesian (adding 90deg to Omega, which is E of N)
    x, y, z = el2xv(a, e, i, peri, node_north+90, f, degrees=True)
#    pa = ( np.rad2deg( np.arctan2(y, x) ) - 90 ) % 360
#    r = np.abs([x + 1j * y])

    return [x, y, z]


def binary_ephemeris(a, e, i, peri, node_north, epoch, q,
                     primary=False, anomaly=None, t=None, period=None):
    '''Return binary ephemerides.
    
    Values are simply passed to orbit_ephemeris and scaled in size.
    
    Parameters
    ----------
    a : float
        Semi-major axis.
    e : float
        Eccentricity.
    i : float
        Inclination.
    peri : float
        Argument of pericenter, in degrees.
    node_north : float
        Longitude of ascending node, East of North, in degrees.
    epoch : ndarray
        List of times to return positions.
    q : float
         Mass ratio m2/m1 of binary.
    primary : bool, optional
        Return primary position, instead of secondary.
    anomaly : float, optional
        True anomaly, must be given if T and period are not.
    t : float
        Time of pericenter passage in years, must be given if anomaly is not.
    period : float
        Orbital period in years, must be given if anomaly is not.
    '''

    b = sky_orbit(*[a,e,i,peri,node_north,epoch],
                  anomaly=anomaly, t=t, period=period)
    a = sky_orbit(*[a,e,i,peri,node_north,epoch],
                  anomaly=anomaly, t=t, period=period, primary=True)
    bary_b = 1 / (1+q)
    bary_a = q / (1+q)
    for i in [0,1,2]:
        a[i] *= bary_a
        b[i] *= bary_b

    return a, b


def convfm(f_in, ecc):
    '''Convert true to mean anomaly.
    
    Parameters
    ----------
    f_in : float or ndarray
        True anomaly.
    ecc : float or ndarray
        Eccentricity
    '''
    tf2 = np.tan(0.5*f_in)
    fact = np.sqrt( (1.0+ecc) / (1.0-ecc) )
    bige = 2 * np.arctan2(tf2, fact)
    bigm = bige - ecc * np.sin(bige)
    
    return bigm


def convmf(m_in, e_in):
    """Convert array of mean to true anomaly (for single e).
        
    From Vallado
    
    .. todo: tidy and include other orbit cases
    """
    
    m = np.array(m_in) % (2. * np.pi)
    numiter = 50
    small = 0.00000001
    if e_in > small:
        
        ecc = e_in * 1.0
        
        #       ;; /* ------------  initial guess ------------- */
        e0 = m + ecc
        lo = np.logical_or( (m < 0.0) & (m > -np.pi), m > np.pi)
        e0[lo] = m[lo] - ecc
        
        ktr = 1
        e1  = e0 + (m - e0 + ecc * np.sin(e0)) / (1.0 - ecc * np.cos(e0))
        while (np.max(np.abs(e1 - e0)) > small) & (ktr <= numiter):
            ktr += 1
            do = np.abs(e1 - e0) > small
            e0[do] = e1[do]
            e1[do] = e0[do] + (m[do] - e0[do] + ecc * np.sin(e0[do])) / (1.0 - ecc * np.cos(e0[do]))
        
        #       ;; /* ---------  find true anomaly  ----------- */
        sinv = (np.sqrt(1.0 - ecc * ecc) * np.sin(e1)) / (1.0-ecc * np.cos(e1))
        cosv = (np.cos(e1) - ecc) / (1.0 - ecc * np.cos(e1))
        nu   = np.arctan2( sinv, cosv)
    
    else:
        #       ;; /* --------------------- circular --------------------- */
        ktr = 0
        nu  = m
        e0  = m

    if ktr > numiter:
        print('WARNING: convmf did not converge')
    
    return nu


@lru_cache(maxsize=2)
def convmf_lookup(n=200):
    '''Return interpolation object for convmf.'''
    Ms = np.linspace(-np.pi, np.pi, n)
    es = np.linspace(0, 1, n)
    f = np.zeros((n,n))
    for i,m in enumerate(Ms):
        for j,e in enumerate(es):
            tmp = convmf([m],e)[0]
            # some fudges to avoid -pi->pi etc. steps in grid
            if tmp > np.pi:
                tmp -= 2*np.pi
            if i == 0 and tmp == np.pi:
                tmp -= 2*np.pi
            f[i,j] = tmp
            
    return scipy.interpolate.RectBivariateSpline(Ms, es, f)


def convmf_fast(m_in, e_in, n=200):
    '''Convert mean to true anomaly with a lookup table.

    Parameters
    ----------
    m_in : float or ndarray
        Mean anomaly.
    e_in : float or ndarray
        Eccentricity.
    '''
    m = m_in % (2*np.pi)
    m[m>=np.pi] -= 2*np.pi
    convmf_interp = convmf_lookup(n=n)
    return convmf_interp.ev(m, e_in)


def circumbinary_a_crit(e,m1,m2):
    '''Holman & Wiegert 1999.'''
    mu = np.min([m1,m2])/(m1+m2)
    return 1.6 + 5.1*e - 2.22*e**2 + 4.12*mu - 4.27*e*mu - 5.09*mu**2 + 4.61*e**2 * mu**2
