


def Coeffs_And_Sources_For_U(su2d, sp2d, coeffs, conv_w, u2d, uBC_w, vis2d, dp_dx, aw_bndry, vol, viscos, urf_vel):
        
   su2d = su2d - dp_dx * vol
 
   # modify su & sp

   su2d[0,:] += conv_w[0,:] * uBC_w
   sp2d[0,:] -= conv_w[0,:]
   vist = vis2d[0,:,] - viscos
   su2d[0,:] += vist * aw_bndry * uBC_w
   sp2d[0,:] -= vist * aw_bndry

   ap2d = coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2] + coeffs[:, :, 3] - sp2d

   # under-relaxation
   ap2d = ap2d / urf_vel
   su2d = su2d + (1 - urf_vel) * ap2d * u2d

   return su2d, sp2d, ap2d
    
    
    


