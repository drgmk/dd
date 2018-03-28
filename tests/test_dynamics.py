import numpy as np

from .context import dd

def test_xv2el_round_trip():
    n = 10000
    x = np.random.uniform(size=(3,n))
    vx = np.random.uniform(size=(3,n)) * 1000
    el = dd.dynamics.xv2el(1, x[0], x[1], x[2], vx[0], vx[1], vx[2])
    out = dd.dynamics.el2xv(el[0], el[1], el[2], el[3],
                            el[4], el[5], mstar=1)
    for i in [0,1,2]:
        assert(np.allclose(x[i], out[i]))
    for i in [0,1,2]:
        assert(np.allclose(vx[i], out[i+3]))

def test_el2xv_round_trip():
    n = 10000
    ae = np.random.uniform(size=(2,n))
    inc = np.random.uniform(size=n) * np.pi
    ang = np.random.uniform(size=(3,n)) * 2 * np.pi
    xv = dd.dynamics.el2xv(ae[0], ae[1], inc, ang[0], ang[1], ang[2], mstar=1)
    el = dd.dynamics.xv2el(1, xv[0], xv[1], xv[2], xv[3], xv[4], xv[5])
    for i in [0,1]:
        assert(np.allclose(ae[i], el[i]))
    assert(np.allclose(inc, el[2]))
    for i in [0,1,2]:
        assert(np.allclose(ang[i], el[i+3]))
