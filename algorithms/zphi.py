# Additional libraries
import numpy as np

#
#    zphi - ZPHI method for correction of Z and D
#
#    z - Z_h (rays x gates)
#    d - ZDR (rays x gates)
#    p - PhiDP (rays x gates)
#    b - Constant b, 1.02 for X-band
#    dr - Gate size in km
#    alpha - The alpha value to use
#    beta - The alpha value to use
#
def zphi(z, d, p, b=1.02, alpha=0.270, beta=0.0455, verb=0):

    # Make a copy of PhiDP
    pp = np.copy(p)

    # Set PhiDP < 0 to 0 (no correction)
    pp[np.nan_to_num(pp) < 0.0] = 0.0

    # Attenuation
    az = alpha * pp ** b
    ad = beta / alpha * az

    if (verb):
        print('beta / alpha = {:.4f}'.format(beta / alpha))

    # Correction
    zc = z + az
    dc = d + ad

    return zc, dc;
