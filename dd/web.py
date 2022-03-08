import astroquery.simbad
import re
import astropy.units as u
import astropy.coordinates as coord

def get_simbad_main_id(name=None, ra=None, dec=None,
                       ra_unit=u.deg, dec_unit=u.deg,
                       radius=2*u.arcsec):
    '''Return the main ID for an object from simbad.'''

    if name is not None:
        t = astroquery.simbad.Simbad.query_object(name)
    elif ra is not None:
        c = coord.SkyCoord(ra, dec, unit=(ra_unit,dec_unit))
        t = astroquery.simbad.Simbad.query_region(c, radius=radius)

    if t is not None:
        mainid = t['MAIN_ID'][0]
        return re.sub('\s+',' ',mainid)
    else:
        return None
