import numpy as np

def dphi_dx(phi_face_w, phi_face_s, dir_area_wx, dir_area_sx, vol):

   phi_w = phi_face_w[0:-1,:] * dir_area_wx[0:-1,:]
   phi_e = -phi_face_w[1:,:] * dir_area_wx[1:,:]
   phi_s = phi_face_s[:,0:-1] * dir_area_sx[:,0:-1]
   phi_n = -phi_face_s[:,1:] * dir_area_sx[:,1:]

   return (phi_w + phi_e + phi_s + phi_n) / vol


def dphi_dy(phi_face_w, phi_face_s, dir_area_wy, dir_area_sy, vol):

   phi_w = phi_face_w[0:-1,:] * dir_area_wy[0:-1,:]
   phi_e = -phi_face_w[1:,:] * dir_area_wy[1:,:]
   phi_s = phi_face_s[:,0:-1] * dir_area_sy[:,0:-1]
   phi_n = -phi_face_s[:,1:] * dir_area_sy[:,1:]

   return (phi_w + phi_e + phi_s + phi_n) / vol



def modify_outlet(conv_w, F_area_w):

   # inlet
   flow_in = np.sum(conv_w[0,:])
   flow_out = np.sum(conv_w[-1,:])
   area_out = np.sum(F_area_w[-1,:])

   uinc = (flow_in - flow_out) / area_out
   ares = F_area_w[-1,:]
   conv_w[-1,:] = conv_w[-1,:] + uinc * ares

   flow_out_new = np.sum(conv_w[-1,:])

   return conv_w



def vist_kom(vis2d, k2d, omega2d, viscos, urfvis):

   visold = vis2d
   vis2d = k2d / omega2d + viscos

   # modify viscosity
   #vis2d = modify_vis(vis2d)

   # under-relax viscosity
   vis2d = urfvis * vis2d + (1 - urfvis) * visold

   return vis2d







