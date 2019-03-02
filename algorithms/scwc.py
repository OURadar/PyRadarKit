# Additional libraries
import numpy as np
import scipy as sp
import scipy.signal

#
#    scwc - Self Consistent With Constraints
#
#    s - SNR (rays x gates)
#    z - Z_h (rays x gates)
#    p - PhiDP (rays x gates)
#    r - RhoHV (rays x gates)
#    b - Constant b, 1.02 for X-band
#    w - Window length for smoothing PhiDP
#    dr - Gate size in km
#    go - Gate origin to process (recommend 20)
#    st - SNR threshold for data selection
#    rt - RhoHV threshold for data selection
#    vt - Variance threshold for PhiDP qualification (recommend < 100.0)
#    ct - Count threshold to consider the range (r0, rm) to be valid
#    alpha - Search range of alpha values
#
def scwc(s, z, p, r,
         b=1.02, w=20, dr=0.03, go=20,
         st=3.0, rt=0.85, vt=20.0, ct=50,
         alpha=np.arange(0.20, 0.51, 0.01), verb=0):

    # Some constants for convenient coding later
    ray_count = s.shape[0]
    gate_count = s.shape[1]
    alpha_count = len(alpha)
    
    # Index of azimuth
    ia = np.arange(ray_count)

    # Reflectivity in linear unit
    zl = 10.0 ** (0.1 * z)
    zl = np.nan_to_num(zl)

    # Quality mask
    mq = np.logical_and(np.nan_to_num(s) > st, np.nan_to_num(r) > rt)

    # A copy of PhiDP with NAN set to 0.0
    pz = np.nan_to_num(np.copy(p))

    # Smoothing
    ww = np.ones(w) / w;
    ps = sp.signal.lfilter(ww, 1.0, pz)

    # Compute local variance: VAR(X) = E{X ** 2} + E{x} ** 2; set outside mq to some large value
    p_var = sp.signal.lfilter(ww, 1.0, pz ** 2) - ps ** 2
    p_var[~mq] = 1000.0

    # Compute local slope
    p_slope = sp.signal.lfilter(ww, 1.0, np.diff(p))

    # Statistical mask
    ms = np.logical_and(mq, p_var < vt)
    ms[:, :-1] = np.logical_and(ms[:, :-1], np.abs(np.nan_to_num(p_slope, 0.0)) < 0.5)
    ms_count = np.sum(ms, axis=1)

    # Data bounds
    r0 = np.argmax(ms[:, go:], axis=1) + go
    rm = p.shape[1] - np.argmax(ms[:, :go:-1], axis=1) - 1

    # Construct edge bound - the index path to use PhiDP
    ah = np.zeros((*ps.shape, alpha_count))
    edge = np.zeros(ms.shape, dtype=bool)
    deltaPhi = np.zeros(ray_count)
    for i, s, e, c in zip(ia, r0, rm, ms_count):
        if c > ct:
            # Only use the path index if the length > 50 cells
            edge[i, s:e] = True;
            deltaPhi[i] = ps[i, e] - ps[i, s]

    # Mask out the Z and smoothed PhiDP values outside (r0, rm)
    zl[~edge] = 0.0

    # Z ** b
    zb = zl ** b;

    # I(r; rm) is a function of r, integrate Z from r to rm
    ir = 0.46 * b * np.cumsum(zb[:, ::-1], axis=1)[:, ::-1] * dr
    ir0 = np.array([x[i] for x, i in zip(ir, r0)])

    # The common term in size of (360, 1, alpha_count)
    tenPowerSomethingMinusOne = 10.0 ** (0.1 * b * np.outer(deltaPhi, alpha).reshape((ps.shape[0], 1, -1)))

    # Repeat Z for all combinations of r and alpha
    zb_big = np.repeat(np.expand_dims(zb, 2), alpha_count, axis=2)
    ir_big = np.repeat(np.expand_dims(ir, 2), alpha_count, axis=2)
    ps_big = np.repeat(np.expand_dims(ps, 2), alpha_count, axis=2)

    # I(r0; rm) is the same for one ray, all gates, all alpha values
    ir0_big = np.repeat(ir0, ps.shape[1] * alpha_count).reshape((*ps.shape, alpha_count))

    # Eq (15) for all (r; rm) so that ir[x] = 0.46 b int_x^rm (z ** b) dr
    num = zb_big * tenPowerSomethingMinusOne
    den = (ir0_big + tenPowerSomethingMinusOne * ir_big)
    den[den == 0.0] = 1.0
    ah_big = num / den

    # Construct PhiDP for all alpha values
    alpha_big = np.outer(np.ones(ps.shape), alpha).reshape((*ps.shape, alpha_count))
    pc_big = 2.0 * np.cumsum(ah_big, axis=1) / alpha_big * dr

    # Pick the best alpha
    err = np.sum(np.abs(ps_big - pc_big), axis=(0, 1))
    alpha_idx = np.argmin(err)

    if (verb):
        print('Best alpha @ {} / {} -> {:.2f}'.format(alpha_idx, alpha_count, alpha[alpha_idx]))

    # Use the original PhiDP values
    pp = np.copy(p)
    mp = np.nan_to_num(pp) <= 0.0
    pp[mp] = 0.0

    at = alpha[alpha_idx] * pp ** b

    return z + at;
