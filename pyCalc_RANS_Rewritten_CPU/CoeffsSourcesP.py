import numpy as np

# Computes sources, coefficients and under-relaxation for pp2d

def Coeffs_And_Sources_For_P(pp2d, ap2d_vel, conv_w, conv_s, ni, nj, fx, fy, dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, urf_vel):
       
   coeffs = np.zeros((ni, nj, 5))

   apw = np.zeros((ni + 1, nj))
   aps = np.zeros((ni, nj + 1))

   pp2d = 0
   
   # simplec: multiply ap by (1-urf)
   ap2d_vel = np.maximum(ap2d_vel * (1 - urf_vel), 1e-20)
   
   # West Face
   apw[0:-1,:] = fx * ap2d_vel + (1 - fx) * np.roll(ap2d_vel, 1, axis=0)
   apw[0,:] = 1e-20
   dw = dir_area_wx**2 + dir_area_wy**2

   coeffs[:,:,1] = dw[0:-1,:] / apw[0:-1,:]

   # East
   coeffs[:,:,0] = np.roll(coeffs[:,:,1], -1, axis=0)

   # South Face
   aps[:,0:-1] = fy * ap2d_vel + (1 - fy) * np.roll(ap2d_vel, 1, axis=1)
   aps[:,0] = 1e-20
   ds = dir_area_sx**2 + dir_area_sy**2

   coeffs[:,:,3] = ds[:,0:-1] / aps[:,0:-1]

   # North
   coeffs[:,:,2] = np.roll(coeffs[:,:,3], -1, axis=1)

   # Fix values on the boundary

   # East
   coeffs[-1,:, 0] = 0
   # West
   coeffs[0,:, 1] = 0
   # North
   coeffs[:,-1, 2] = 0
   # South
   coeffs[:, 0, 3] = 0

   coeffs[:,:,4] = coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2] + coeffs[:, :, 3]
    
   # Continuity Error
   su2d = conv_w[0:-1,:] - conv_w[1:,:] + conv_s[:,0:-1] - conv_s[:,1:]


   # set pp2d = 0 in [0,0] tp make it non-singular
   coeffs[0, 0, 0:4] = 0
   coeffs[0, 0, 4] = 1

   return coeffs, su2d 

