import Constants as const
from Utils import dphi_dx, dphi_dy

def Correct_Flow_Variables(u2d, v2d, p2d, conv_w, conv_s, aw2d, as2d, pp2d, dpp_dx, dpp_dy, ap2d_vel, vol):

   # Correct Convections
    
   # West face 
   conv_w[1:-1,:] = conv_w[1:-1,:] + aw2d[0:-1,:]*(pp2d[1:,:] - pp2d[0:-1,:])

   # South face
   conv_s[:,1:-1] = conv_s[:,1:-1] + as2d[:,0:-1]*(pp2d[:,1:] - pp2d[:,0:-1])

   # Correct p
   p2d = p2d + const.urf_p*(pp2d - pp2d[0,0])

   # Correct Velocities 

   # compute pressure correcion at faces (N.B. p_bc_west,, ... are not used since we impose Neumann b.c., everywhere)

   u2d = u2d - dpp_dx * vol / ap2d_vel
   v2d = v2d - dpp_dy * vol / ap2d_vel


   return u2d, v2d, p2d, conv_w, conv_s

