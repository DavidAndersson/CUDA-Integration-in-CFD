
def Coeffs_And_Sources_Omega(omega2d, omegaBC_w, aw_bndry, conv_w, su2d, sp2d, coeffs, vis2d, vol, gen, urf_omega, viscos, c_omega_1, c_omega_2, urf_vel):
   
    # Production term
    su2d = su2d + c_omega_1 * gen * vol

    # Dissipation term
    sp2d = sp2d - c_omega_2 * omega2d * vol

    # modify su & sp
    su2d[0,:] = su2d[0,:] + conv_w[0,:] * omegaBC_w
    sp2d[0,:] = sp2d[0,:] - conv_w[0,:]
    vist = vis2d[0,:,] - viscos
    su2d[0,:] = su2d[0,:] + vist * aw_bndry * omegaBC_w
    sp2d[0,:] = sp2d[0,:] - vist * aw_bndry

    ap2d =  coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2] + coeffs[:, :, 3] - sp2d

    # under-relaxation
    ap2d = ap2d / urf_vel
    su2d = su2d + (1 - urf_omega) * ap2d * omega2d
    
    return su2d, sp2d, ap2d

