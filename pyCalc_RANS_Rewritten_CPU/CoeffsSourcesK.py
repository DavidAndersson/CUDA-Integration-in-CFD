import numpy as np

def Coeffs_And_Sources_For_k(k2d, kBC_w, aw_bndry, omega2d, conv_w, dudx, dudy, dvdx, dvdy, su2d, sp2d, coeffs, vis2d, vol, viscos, cmu, urf_k):

    # production term
    gen = 2 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2

    vist = np.maximum(vis2d - viscos, 1e-10)
    su2d = su2d + vist * gen * vol

    sp2d = sp2d - cmu * omega2d * vol

    # modify su & sp
    su2d[0,:] = su2d[0,:] + conv_w[0,:] * kBC_w
    sp2d[0,:] = sp2d[0,:] - conv_w[0,:]
    vist = vis2d[0,:,] - viscos
    su2d[0,:] = su2d[0,:] + vist * aw_bndry * kBC_w
    sp2d[0,:] = sp2d[0,:] - vist * aw_bndry

    ap2d =  coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2] + coeffs[:, :, 3] - sp2d

    # under-relaxation
    ap2d = ap2d / urf_k
    su2d = su2d + (1 - urf_k) * ap2d * k2d

    return su2d, sp2d, gen, ap2d