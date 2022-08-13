import numpy as np

# Computes all the coefficents

def Coeff(coeffs, conv_w, conv_s, vis2d, prand, scheme_local, ni, nj, fx, fy, F_area_w, F_area_s, vol, viscos):

   vis_w = np.zeros((ni + 1, nj))
   vis_s = np.zeros((ni, nj + 1))

   vis_turb = (vis2d - viscos) / prand

   # Interpolate viscosity to faces
   vis_w[0:-1, :] = fx * vis_turb + (1 - fx) * np.roll(vis_turb, 1, axis=0) + viscos
   vis_s[:, 0:-1] = fy * vis_turb + (1 - fy) * np.roll(vis_turb, 1, axis=1) + viscos

   vol_w = np.ones((ni + 1, nj)) * 1e-10
   vol_s = np.ones((ni, nj + 1)) * 1e-10
   

   # (vol[i,j] - vol[i+1, j]) / 2 -> Is that right? Should it not be roll -1
   vol_w[1:, :] = 0.5 * np.roll(vol, -1, axis=0) + 0.5 * vol
   vol_s[:, 1:] = 0.5 * np.roll(vol, -1, axis=1) + 0.5 * vol
   
   diff_w = vis_w[0:-1, :] * F_area_w[0:-1, :]**2 / vol_w[0:-1, :]
   diff_s = vis_s[:, 0:-1] * F_area_s[:, 0:-1]**2 / vol_s[:, 0:-1]

   if scheme_local == 1:
        
      # East
      coeffs[:,:, 0] = np.maximum(-conv_w[1:, :], np.roll(diff_w, -1, axis=0) - np.roll(fx, -1, axis=0) * conv_w[1:, :])
      coeffs[:,:, 0] = np.maximum(coeffs[:,:, 0], 0)

      # West
      coeffs[:,:, 1] = np.maximum(conv_w[0:-1, :], diff_w + (1 - fx) * conv_w[0:-1, :])
      coeffs[:,:, 1] = np.maximum(coeffs[:,:, 1], 0)

      # North
      coeffs[:,:, 2] = np.maximum(-conv_s[:, 1:], np.roll(diff_s, -1, axis=1) - np.roll(fy, -1, axis=1) * conv_s[:, 1:]) 
      coeffs[:,:, 2] = np.maximum(coeffs[:,:, 2], 0)

      # South
      coeffs[:,:, 3] = np.maximum(conv_s[:, 0:-1], diff_s+(1 - fy) * conv_s[:, 0:-1])
      coeffs[:,:, 3] = np.maximum(coeffs[:,:, 3], 0)

      
   if scheme_local == 2:
      # East
      coeffs[:,:, 0] = np.roll(diff_w, -1, axis=0) - np.roll(fx, -1, axis=0) * conv_w[1:,:]
      # West
      coeffs[:,:, 1] = diff_w + (1 - fx) * conv_w[0:-1,:]
      # North
      coeffs[:,:, 2] = np.roll(diff_s, -1, axis=1) - np.roll(fy, -1, axis=1) * conv_s[:,1:]
      # South
      coeffs[:,:, 3] = diff_s + (1 - fy) * conv_s[:,0:-1]
      
   # Fix the boundary values
   coeffs[-1,:, 0] = 0
   coeffs[0,:, 1] = 0
   coeffs[:, -1, 2] = 0
   coeffs[:,0, 3] = 0
   
   return coeffs