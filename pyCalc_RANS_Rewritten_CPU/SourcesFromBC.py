import numpy as np

def Sources_From_BC(phiBC_e, phiBC_w, phiBC_n, phiBC_s, phiBC_Types, a_e, a_w, a_n, a_s, ni, nj, viscos):
    
    # Note that the coefficients in the parameters are BOUNDARY coefficients

    su2d = np.zeros((ni, nj))
    sp2d = np.zeros((ni, nj))

    # East
    if phiBC_Types[0] == 'd':    
        sp2d[-1, :] -= viscos * a_e
        su2d[-1, :] +=  viscos * a_e * phiBC_e

    # West
    if phiBC_Types[1] == 'd':    
        sp2d[0, :] -= viscos * a_w
        su2d[0, :] += viscos * a_w * phiBC_w

    # North
    if phiBC_Types[2] == 'd':    
        sp2d[:, -1] -= viscos * a_n
        su2d[:, -1] += viscos * a_n * phiBC_n

    # South
    if phiBC_Types[3] == 'd':
       sp2d[:,0] -= viscos * a_s
       su2d[:,0] += viscos * a_s * phiBC_s

    
    return su2d, sp2d