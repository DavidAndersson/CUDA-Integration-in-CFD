from numba import cuda
import numpy as np
import time
import math
import Residuals as res
from Solve_2d import Solve_2d
from SetupFile import Setup
from Generate_Mesh import Generate_Mesh

ProgStart = time.time()

#### User Input #####

# Total Domain Size
ni = 512
nj = 512

# Block Size
TPB_X = 32
TPB_Y = 32

########################

# Will not generate if a mesh of that size already exists. 
xFile, yFile = Generate_Mesh(ni, nj)

# Shared memory size -> Two Way access and One Way access
Sh_Size_TW = (TPB_X + 2, TPB_Y + 2)
Sh_Size_OW = (TPB_X + 1, TPB_Y + 1)
Sh_Size_THW = (TPB_X + 3, TPB_Y + 3) # Three Way access (two up, one back)
Sh_Coeff_Size_OW = (TPB_X + 1, TPB_Y + 1, 4)


Setup(xFile, yFile)
from Constants import *
from FlowData import *
from GeometricData import *
from CopyToDevice import *

viscos = np.float32(viscos)
urf_vel = np.float32(urf_vel) 


def main():
    
    ##########   CUDA DATA  ####################
    ThreadsPerBlock = (TPB_X, TPB_Y)
    
    blocks_in_grid_x = int(math.ceil(ni / TPB_X))
    blocks_in_grid_y = int(math.ceil(nj / TPB_Y))
    
    BlocksInGrid = (blocks_in_grid_x, blocks_in_grid_y)
    
    global u2d
    global v2d
    
    iterStart = time.time() 
    Total_Iter_Time = 0
    for iteration in range(maxit):

        iterTime = time.time()
         
        # Get coefficients for u and v (except for a_p)
        Coeff[BlocksInGrid, ThreadsPerBlock](coeffs_UV, conv_w, conv_s, vis2d, scheme, ni, nj, fx, fy, F_area_w, F_area_s, vol)              
        
        Clear_Variable_2d[BlocksInGrid, ThreadsPerBlock](su2d)
        Clear_Variable_2d[BlocksInGrid, ThreadsPerBlock](sp2d)
    
        # u2d
        Sources_From_BC[BlocksInGrid, ThreadsPerBlock](su2d, sp2d, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos)   
        dphi_dx[BlocksInGrid, ThreadsPerBlock](dp_dx, pface_w, pface_s, dir_area_wx, dir_area_sx, vol, ni, nj)                                                    
        Coeffs_And_Sources_For_U[BlocksInGrid, ThreadsPerBlock](su2d, sp2d, coeffs_UV, u2d, dp_dx, vol, urf_vel, ni, nj)                                            
        
        # Solve u2d
        #Host_u2d = u2d.copy_to_host()	
        #Host_coeff = coeffs_UV.copy_to_host()
        #Host_su2d = su2d.copy_to_host()			    
        #u2d, res.u = Solve_2d(Host_u2d, Host_coeff, Host_su2d, convergence_limit_u, nsweep_vel, solver_vel, ni, nj)
        #u2d = cuda.to_device(np.float32(u2d)) 
    
        Clear_Variable_2d[BlocksInGrid, ThreadsPerBlock](su2d)
        Clear_Variable_2d[BlocksInGrid, ThreadsPerBlock](sp2d)
    
        # v2d
        Sources_From_BC[BlocksInGrid, ThreadsPerBlock](su2d, sp2d, vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos)  
        dphi_dy[BlocksInGrid, ThreadsPerBlock](dp_dy, pface_w, pface_s, dir_area_wy, dir_area_sy, vol, ni, nj)                                                      
        Coeffs_And_Sources_For_V[BlocksInGrid, ThreadsPerBlock](su2d, sp2d, coeffs_UV, v2d, dp_dy, vol, urf_vel, ni, nj)                                            
        
        # Solve v2d
        #Host_v2d = v2d.copy_to_host()
        #Host_coeff = coeffs_UV.copy_to_host()
        #Host_su2d = su2d.copy_to_host()
        #v2d, res.v = Solve_2d(Host_v2d, Host_coeff, Host_su2d, convergence_limit_v, nsweep_vel, solver_vel, ni, nj)
        #v2d = cuda.to_device(np.float32(v2d))
        
        Clear_Variable_2d[BlocksInGrid, ThreadsPerBlock](su2d)
        Clear_Variable_3d[BlocksInGrid, ThreadsPerBlock](coeffs_pp)
    
        # pp2d
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](uface_w, uface_s, u2d, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ni, nj, fx, fy) 
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](vface_w, vface_s, v2d, vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ni, nj, fx, fy)                                  
    
        ConvectionCorrection[BlocksInGrid, ThreadsPerBlock](velCorr_w, velCorr_s, p2d, coeffs_UV, F_area_w, F_area_s, fx, fy, ni, nj)
        
        Convection_CUDA[BlocksInGrid, ThreadsPerBlock](conv_w, conv_s, velCorr_w, velCorr_s, uface_w, uface_s, vface_w, vface_s, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, \
                                                       vBC_e, vBC_w, vBC_n, vBC_s, ni, nj, dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, F_area_w, F_area_s)
    
        Coeffs_And_Sources_For_P[BlocksInGrid, ThreadsPerBlock](coeffs_pp, su2d, coeffs_UV, conv_w, conv_s, ni, nj, fx, fy, \
                                                                dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, urf_vel)
    
        # Solve pp2d
        #pp2d = np.zeros((ni, nj)) 
        #Host_coeff = coeffs_pp.copy_to_host()
        #Host_su2d = su2d.copy_to_host()
        #Host_pp2d, res.p = Solve_2d(pp2d, Host_coeff, Host_su2d, convergence_limit_pp, nsweep_pp, solver_pp, ni, nj)
        #res.pp = abs(np.sum(Host_su2d))
        #pp2d = cuda.to_device(Host_pp2d)
        
        # Correct u, v, p
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](ppface_w, ppface_s, pp2d, pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types, ni, nj, fx, fy)         
        dphi_dx[BlocksInGrid, ThreadsPerBlock](dpp_dx, ppface_w, ppface_s, dir_area_wx, dir_area_sx, vol, ni, nj)        
        dphi_dy[BlocksInGrid, ThreadsPerBlock](dpp_dy, ppface_w, ppface_s, dir_area_wy, dir_area_sy, vol, ni, nj)    
    
        Correct_Flow_Variables[BlocksInGrid, ThreadsPerBlock](u2d, v2d, p2d, conv_w, conv_s, coeffs_pp, pp2d,\
                                                              dpp_dx, dpp_dy, coeffs_UV, vol, ni, nj, urf_p)       
    
        
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](uface_w, uface_s, u2d, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ni, nj, fx, fy)   
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](vface_w, vface_s, v2d, vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ni, nj, fx, fy)  
        Face_Phi_CUDA[BlocksInGrid, ThreadsPerBlock](pface_w, pface_s, p2d, pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types, ni, nj, fx, fy) 
    
        # scale residuals
        res.u /= resnorm_vel
        res.v /= resnorm_vel
        res.p /= resnorm_p
    
        resmax = np.max([res.u, res.v, res.p])
        
        if iteration > 0:
            Total_Iter_Time += time.time() - iterTime
        
        print("________________________________________________________________")
        print("\nIteration: " + str(iteration) + "\t Iteration Time: " + str(time.time() - iterTime))
        print("\nResiduals: ")
        print("max: " + str("{:e}".format(resmax)) + 
              "\t u: " + str("{:e}".format(res.u)) + 
              "\t v: " + str("{:e}".format(res.v)) + 
              "\t pp: " + str("{:e}".format(res.p)))
        
    
        maxU = np.zeros((2,2), dtype=np.float32)
        maxV = np.zeros((2,2), dtype=np.float32)
        FindMax[BlocksInGrid, ThreadsPerBlock](maxU, u2d)
        FindMax[BlocksInGrid, ThreadsPerBlock](maxV, v2d)
         
        
        print("\nData: ")
        print("U max: " + "{:e}".format(maxU[0,0]) + 
              "\t V max: " + "{:e}".format(maxV[0,0]))
    
        #if resmax < sormax: 
           #break
            

    print("\n\nIteration duration: " + str(time.time() - iterStart) + "s")
    print("Average iteration duration: " + str(Total_Iter_Time / maxit) + "s")
    print("Prgram duration: " + str(time.time() - ProgStart) + "s")



#%% Kernels

@cuda.jit
def Coeff(coeffs, conv_w, conv_s, vis2d, scheme_local, ni, nj, fx, fy, F_area_w, F_area_s, vol):

   i, j = cuda.grid(2) 

   # Find adjacent global inidices, and do global bounds check
   i_left  = max( i-1, 0 )
   i_right = min( i+1, ni - 1 )
   j_down  = max( j-1, 0 )
   j_up    = min( j+1, nj - 1 )

   if i < ni and j < nj:

       # Pre-fetch values to hide latency
       volume = vol[i, j]
       volume_left = vol[i_left, j]
       volume_right = vol[i_right, j]
       volume_down = vol[i, j_down]
       volume_up = vol[i, j_up]
       vis = vis2d[i, j]
       conv_W = conv_w[i, j] 
       conv_w_right = conv_w[i + 1, j]
       conv_S = conv_s[i, j]
       conv_s_up = conv_s[i, j + 1]
       F_area_W = F_area_w[i, j]
       F_area_w_right = F_area_w[i + 1, j]
       F_area_S = F_area_s[i, j]
       F_area_s_up = F_area_s[i, j + 1]
       f_x = fx[i, j]
       f_x_right = fx[i_right, j]
       f_y = fy[i, j]
       f_y_up = fy[i, j_up]
       

       # Define shared arrays
       Sh_vol = cuda.shared.array(shape=Sh_Size_TW, dtype = np.float32)

       Sh_vol_w = cuda.shared.array(shape=Sh_Size_OW, dtype = np.float32)
       Sh_vol_s = cuda.shared.array(shape=Sh_Size_OW, dtype = np.float32)
       Sh_diff_w = cuda.shared.array(shape=Sh_Size_OW, dtype = np.float32)
       Sh_diff_s = cuda.shared.array(shape=Sh_Size_OW, dtype = np.float32)
       
       tx = cuda.threadIdx.x
       ty = cuda.threadIdx.y
       
       # Shared-array index
       six = tx + 1
       siy = ty + 1

       # Load in main parts of shared arrays -> Needs to add boundary values afterwards
       Sh_vol[six, siy] = volume

       # Fill boundary values

       # Left
       if tx == 0:
           Sh_vol[0, siy] = volume_left
       # Right
       if tx == bw-1:
           Sh_vol[six + 1, siy] = volume_right

       # Down
       if ty == 0:
           Sh_vol[six, 0] = volume_down
       # Up
       if ty == bh-1:
           Sh_vol[six, siy + 1] = volume_up
           
       cuda.syncthreads()

       # Interpolate volume to faces
       Sh_vol_w[tx, ty] = 0.5 * Sh_vol[six - 1, siy] + 0.5 * Sh_vol[six, siy] if i > 0 else 1e-10
       Sh_vol_s[tx, ty] = 0.5 * Sh_vol[six, siy - 1] + 0.5 * Sh_vol[six, siy] if j > 0 else 1e-10
       
       # Fill boundary values

       # Right (East)
       if tx == bw-1:
           Sh_vol_w[tx + 1, ty] = 0.5 * Sh_vol[six, siy] + 0.5 * Sh_vol[six + 1, siy]

       # Up (North)
       if ty == bh-1:
           Sh_vol_s[tx, ty + 1] = 0.5 * Sh_vol[six, siy] + 0.5 * Sh_vol[six, siy + 1]

       cuda.syncthreads()

       # Compute diffusion at the faces
       Sh_diff_w[tx, ty] = vis * F_area_W**2 / Sh_vol_w[tx, ty] 
       Sh_diff_s[tx, ty] = vis * F_area_S**2 / Sh_vol_s[tx, ty] 

       # Fill in the extra cells
       if tx == bw - 1:
           Sh_diff_w[tx + 1, ty] = vis * F_area_w_right**2 / Sh_vol_w[tx + 1, ty] 

       if ty == bh - 1:
           Sh_diff_s[tx, ty + 1] = vis * F_area_s_up**2 / Sh_vol_s[tx, ty + 1] 

       cuda.syncthreads()     

       # Compute the coefficients. Scheme 1 - means Hybrid. Scheme 2 is central upwind. 
       # chars seem to not be compatible with CUDA kernels for some reason.   
   
       if scheme_local == 1:
        
          # East
          if i < ni - 1:
              coeffs[i, j, 0] = max( -conv_w_right,  Sh_diff_w[tx + 1, ty] - f_x_right * conv_w_right,     0 )
          else:
              coeffs[i, j, 0] = 0

          # West
          if i > 0:
              coeffs[i, j, 1] = max(  conv_W,   Sh_diff_w[tx, ty] + (1 - f_x) * conv_W,   0 )
          else:
              coeffs[i, j, 1] = 0

          # North
          if j < nj - 1:
              coeffs[i, j, 2] = max( -conv_s_up,  Sh_diff_s[tx, ty + 1] - f_y_up * conv_s_up,     0 ) 
          else:
              coeffs[i, j, 2] = 0  
          
          # South
          if j > 0:
              coeffs[i, j, 3] = max(  conv_S,   Sh_diff_s[tx, ty] + (1 - f_y) * conv_S,     0 )
          else:
              coeffs[i, j, 3] = 0

     
       elif scheme_local == 2:

          # East
          if i > 0:
              coeffs[i, j, 0] = Sh_diff_w[tx + 1, ty] - f_x * conv_W
          else: 
              coeffs[i, j, 0] = 0
          
          # West
          if i < ni - 1:
              coeffs[i, j, 1] = Sh_diff_w[tx, ty] + (1 - f_x) * conv_W
          else: 
              coeffs[i, j, 1] = 0
                    
          # North 
          if j > 0:
              coeffs[i, j, 2] = Sh_diff_s[tx, ty + 1] - f_y * conv_S
          else: 
              coeffs[i, j, 2] = 0
          
          # South
          if j < nj - 1:
              coeffs[i, j, 3] = Sh_diff_s[tx, ty] + (1 - f_y) * conv_S
          else: 
              coeffs[i, j, 3] = 0


@cuda.jit
def Clear_Variable_2d(Phi):

    i, j = cuda.grid(2)

    Phi[i, j] = 0


@cuda.jit
def Clear_Variable_3d(Phi):

    i, j = cuda.grid(2)

    for k in range(Phi.shape[2]):
        Phi[i, j, k] = 0


@cuda.jit
def Sources_From_BC(su2d, sp2d, phiBC_e, phiBC_w, phiBC_n, phiBC_s, phiBC_Types, a_e, a_w, a_n, a_s, ni, nj, viscos):
    
    # Note that the coefficients in the parameters are BOUNDARY coefficients

    # Boundary Types
    # 1. Dirichlet
    # 2. Neumann

    i, j = cuda.grid(2)

    if i < ni and j < nj:

        a_e = a_e[j]
        BC_e = phiBC_e[j]
        a_w = a_w[j]
        BC_w = phiBC_w[j]
        a_n = a_n[i]
        BC_n = phiBC_n[i]
        a_s = a_s[i]
        BC_s = phiBC_s[i]
        
        # East
        if i == ni - 1 and phiBC_Types[0] == 1:
            sp2d[i, j] -= viscos * a_e
            su2d[i, j] += viscos * a_e * BC_e

        # West
        if i == 0 and phiBC_Types[0] == 1:
            sp2d[i, j] -= viscos * a_w
            su2d[i, j] += viscos * a_w * BC_w

        # North
        if j == nj - 1 and phiBC_Types[0] == 1: 
            sp2d[i, j] -= viscos * a_n
            su2d[i, j] += viscos * a_n * BC_n

        # South
        if j == 0 and phiBC_Types[0] == 1:
            sp2d[i, j] -= viscos * a_s
            su2d[i, j] += viscos * a_s * BC_s


@cuda.jit
def dphi_dx(dphi_dx, phi_face_w, phi_face_s, dir_area_wx, dir_area_sx, vol, ni, nj):

    # Important note: 
    # phi_face_w and dir_area_wx are (ni+1, nj) 
    # phi_face_s and dir_area_sx are (ni, nj+1)

    i, j = cuda.grid(2)

    if i < ni and j < nj:

        volume = vol[i, j]

        # East
        phi_e = -phi_face_w[i+1, j] * dir_area_wx[i+1, j]

        # West
        phi_w = phi_face_w[i, j] * dir_area_wx[i, j]

        # North
        phi_n = -phi_face_s[i, j+1] * dir_area_sx[i, j+1]

        # South
        phi_s = phi_face_s[i, j] * dir_area_sx[i, j]

        dphi_dx[i, j] = (phi_w + phi_e + phi_s + phi_n) / volume


@cuda.jit
def dphi_dy(dphi_dy, phi_face_w, phi_face_s, dir_area_wy, dir_area_sy, vol, ni, nj):

    # Important note: 
    # phi_face_w and dir_area_wx is (ni+1, nj) 
    # phi_face_s and dir_area_sx is (ni, nj+1)

    i, j = cuda.grid(2)

    if i < ni and j < nj:

       volume = vol[i, j]

       # East
       phi_e = -phi_face_w[i+1, j] * dir_area_wy[i+1, j]

       # West
       phi_w = phi_face_w[i, j] * dir_area_wy[i, j]

       # North
       phi_n = -phi_face_s[i, j+1] * dir_area_sy[i, j+1]

       # South
       phi_s = phi_face_s[i, j] * dir_area_sy[i, j]

       dphi_dy[i, j] = (phi_w + phi_e + phi_s + phi_n) / volume


@cuda.jit
def Coeffs_And_Sources_For_U(su2d, sp2d, coeffs, du2d, dp_dx, vol, urf_vel, ni, nj):

    i, j = cuda.grid(2)

    if i < ni and j < nj:

        u = du2d[i, j]
        
        su2d[i, j] -= dp_dx[i, j] * vol[i, j]
 
        coeffs[i, j, 4] = coeffs[i, j, 0] + coeffs[i, j, 1] + coeffs[i, j, 2] + coeffs[i, j, 3] - sp2d[i, j]

        # under-relaxation
        coeffs[i, j, 4] /= urf_vel
        su2d[i, j] += (1 - urf_vel) * coeffs[i, j, 4] * u


@cuda.jit
def Coeffs_And_Sources_For_V(su2d, sp2d, coeffs, v2d, dp_dy, vol, urf_vel, ni, nj):
       
    i, j = cuda.grid(2)

    if i < ni and j < nj:

        v = v2d[i, j]
        
        su2d[i, j] -= dp_dy[i, j] * vol[i, j]
 
        coeffs[i, j, 4] = coeffs[i, j, 0] + coeffs[i, j, 1] + coeffs[i, j, 2] + coeffs[i, j, 3] - sp2d[i, j]

        # under-relaxation, and assign ap_2d to the coefficent variable
        coeffs[i, j, 4] /= urf_vel
        su2d[i, j] += (1 - urf_vel) * coeffs[i, j, 4] * v


@cuda.jit
def Face_Phi_CUDA(phi_face_w, phi_face_s, phi2d, phiBC_e, phiBC_w, phiBC_n, phiBC_s, phiBC_Types, ni, nj, fx, fy):

    # Can be implemented with shared memory

    i, j = cuda.grid(2)

    i_left = max(i - 1, 0)
    j_down = max(j - 1, 0)

    if i < ni and j < nj:

        f_x = fx[i, j]
        f_y = fy[i, j]
        phi = phi2d[i, j]
        BC_e = phiBC_e[j]
        BC_w = phiBC_w[j]
        BC_n = phiBC_n[i]
        BC_s = phiBC_s[i]

        # Handle all internal cells
    
        phi_face_w[i, j] = f_x * phi + (1 - f_x) * phi2d[i_left, j]
        phi_face_s[i, j] = f_y * phi + (1 - f_y) * phi2d[i, j_down]

        # Handle the boundaries:
        # - If Dirichlet -> apply given values
        # - else Neumann (Homogeneous) -> copy the corresponding value from phi2d

        # Boundary Types
        # 1. Dirichlet
        # 2. Neumann

        # BC Types follow same pattern as coeffs. 0-east ... 3-south

        # East Boundary
        if i == ni - 1:
            if phiBC_Types[0] == 1:
                phi_face_w[-1, j] = BC_e
            else:
                phi_face_w[-1, j] = phi 

        # West Boundary
        elif i == 0:
            if phiBC_Types[1] == 1:
                phi_face_w[0, j] = BC_w
            else:
                phi_face_w[0, j] = phi

        # North Boundary
        if j == nj - 1:
            if phiBC_Types[2] == 1:
                phi_face_s[i, -1] = BC_n
            else:
                phi_face_s[i, -1] = phi

        # South Boundary
        elif j == 0:
            if phiBC_Types[3] == 1:
                phi_face_s[i, 0] = BC_s
            else:
                phi_face_s[i, 0] = phi


@cuda.jit
def ConvectionCorrection(velCorr_w, velCorr_s, p2d, coeffsUV, F_area_w, F_area_s, fx, fy, ni, nj):

   i, j = cuda.grid(2)

   # Find adjacent global inidices, and do global bounds check
   i_left  = max( i - 1, 0 )
   i_left_left = max(i_left - 1, 0)
   i_right = min( i + 1, ni - 1 )
   j_down  = max( j - 1, 0 )
   j_down_down = max(j_down - 1, 0)
   j_up    = min( j + 1, nj - 1 )

   if i < ni and j < nj:

        # Pre-fetch variables to hide latency
        f_x = fx[i, j]
        f_y = fy[i, j]
        a_p = coeffsUV[i, j, 4]

        # Size: Block dims + 3
        Sh_p2d = cuda.shared.array(shape=Sh_Size_THW, dtype = np.float32)
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        six = tx + 2
        siy = ty + 2

        Sh_p2d[six, siy] = p2d[i, j]

        if tx == bw - 1 or i == ni - 1:
            Sh_p2d[six + 1, siy] = p2d[i_right, j]
        if tx == 0:
            Sh_p2d[six - 1, siy] = p2d[i_left, j]
            Sh_p2d[six - 2, siy] = p2d[i_left_left, j]

        if ty == bh - 1 or j == nj - 1:
            Sh_p2d[six, siy + 1] = p2d[i, j_up]
        if ty == 0:
            Sh_p2d[six, siy - 1] = p2d[i, j_down]
            Sh_p2d[six, siy - 2] = p2d[i, j_down_down]

         
        cuda.syncthreads()
        
        # East - West

        dp = Sh_p2d[six + 1, siy] - 3 * Sh_p2d[six, siy] + 3 * Sh_p2d[six - 1, siy] - Sh_p2d[six - 2, siy]
        ap_w = f_x * a_p + (1 - f_x) * coeffsUV[i_left, j, 4]
        
        # boundaries (no corrections) -> east boundary is handled implicitly, as ni + 1 (or index ni) never gets a thread
        if i == 0:
            velCorr_w[i, j] = 0
        else:
            velCorr_w[i, j] = dp * F_area_w[i, j] / ( 4 * ap_w )
        


        # North - South

        dp = Sh_p2d[six, siy + 1] - 3 * Sh_p2d[six, siy] + 3 * Sh_p2d[six, siy - 1] - Sh_p2d[six, siy - 2]
        ap_s = f_y * a_p + (1 - f_y) * coeffsUV[i, j_down, 4]

        # boundaries (no corrections) -> east boundary is handled implicitly, as ni + 1 (or index ni) never gets a thread
        if j == 0:
            velCorr_s[i, j] = 0
        else:
            velCorr_s[i, j] = dp * F_area_s[i, j] / ( 4 * ap_s )


@cuda.jit
def Convection_CUDA(conv_w, conv_s, velCorr_w, velCorr_s, uface_w, uface_s, vface_w, vface_s, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types,\
                    vBC_e, vBC_w, vBC_n, vBC_s, ni, nj, dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, F_area_w, F_area_s):


   i, j = cuda.grid(2)

   if i < ni and j < nj:

        conv_w[i, j] = -uface_w[i, j] * dir_area_wx[i, j] - vface_w[i, j] * dir_area_wy[i, j] + F_area_w[i, j] * velCorr_w[i, j]
        conv_s[i, j] = -uface_s[i, j] * dir_area_sx[i, j] - vface_s[i, j] * dir_area_sy[i, j] + F_area_s[i, j] * velCorr_s[i, j]

        # conv_w and conv_s are ni+1 and nj+1 respectively (Hopefully the variables here are cached)
        if i == ni - 1:
            conv_w[i+1, j] = -uface_w[i+1,j] * dir_area_wx[i+1,j] - vface_w[i+1,j] * dir_area_wy[i+1,j]
        if j == nj - 1:
            conv_s[i, j+1] = -uface_s[i,j+1] * dir_area_sx[i,j+1] - vface_s[i,j+1] * dir_area_sy[i,j+1]   
        
        
        # Boundaries

         # Boundary Types
         # 1. Dirichlet
         # 2. Neumann

        # East
        if uBC_Types[0] == 1 and i == ni - 1:
            conv_w[-1, j] = -uBC_e[j] * dir_area_wx[-1, j] - vBC_e[j] * dir_area_wy[-1, j]

        # West
        elif uBC_Types[1] == 1 and i == 0:
            conv_w[0, j] = -uBC_w[j] * dir_area_wx[i, j] - vBC_w[j] * dir_area_wy[i, j]

        # North
        if uBC_Types[2] == 1 and j == nj - 1:
            conv_s[i, -1] = -uBC_n[i] * dir_area_sx[i, -1] - vBC_n[i] * dir_area_sy[i, -1]

        # South
        elif uBC_Types[3] == 1 and j == 0:
            conv_s[i, 0] = -uBC_s[i] * dir_area_sx[i, j] - vBC_s[i] * dir_area_sy[i, j]
        

@cuda.jit
def Coeffs_And_Sources_For_P(coeffs, su2d, coeffsUV, conv_w, conv_s, ni, nj, fx, fy, dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, urf_vel):
       
    i, j = cuda.grid(2)

    # Find adjacent global inidices, and do global bounds check
    i_left  = max( i-1, 0 )
    i_right = min( i+1, ni - 1 )
    j_down  = max( j-1, 0 )
    j_up    = min( j+1, nj - 1 )

    if i < ni and j < nj:
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        # Shared array index
        six = tx + 1
        siy = ty + 1

        # Pre-fetch some variables to hide latency
        f_x = fx[i, j]
        f_x_right = fx[i_right, j]
        f_y = fy[i, j]
        f_y_up = fy[i, j_up]
        conv_W = conv_w[i, j]
        conv_S = conv_s[i, j]

        Sh_ap2d_vel = cuda.shared.array(shape=Sh_Size_TW, dtype = np.float32)
        Sh_coeff = cuda.shared.array(shape=Sh_Coeff_Size_OW, dtype = np.float32) 

        # simplec: multiply ap by (1-urf)
        Sh_ap2d_vel[six, siy] = max(coeffsUV[i, j, 4] * (1 - urf_vel), 1e-20)

        if tx == 0:
            Sh_ap2d_vel[0, siy] = max(coeffsUV[i_left, j, 4] * (1 - urf_vel), 1e-20)
        if ty == 0:
            Sh_ap2d_vel[six, 0] = max(coeffsUV[i, j_down, 4] * (1 - urf_vel), 1e-20)
        if tx == bw - 1:
            Sh_ap2d_vel[six + 1, siy] = max(coeffsUV[i_right, j, 4] * (1 - urf_vel), 1e-20)
        if ty == bh - 1:
            Sh_ap2d_vel[six, siy + 1] = max(coeffsUV[i, j_up, 4] * (1 - urf_vel), 1e-20)

        # West Face
        if i == 0:
            Sh_coeff[tx, ty, 1] = 0
        else:
            apw = f_x * Sh_ap2d_vel[six, siy] + (1 - f_x) * Sh_ap2d_vel[six - 1, siy]
            dw = dir_area_wx[i, j]**2 + dir_area_wy[i, j]**2
            Sh_coeff[tx, ty, 1] = dw / apw

            if tx == bw - 1:
                apw = f_x_right * Sh_ap2d_vel[six + 1, siy] + (1 - f_x_right) * Sh_ap2d_vel[six, siy]
                dw = dir_area_wx[i_right, j]**2 + dir_area_wy[i_right, j]**2
                Sh_coeff[tx + 1, ty, 1] = dw / apw

        cuda.syncthreads()

        # East
        if i == ni - 1:
            Sh_coeff[tx, ty, 0] = 0
        else:
            Sh_coeff[tx, ty, 0] = Sh_coeff[tx + 1, ty, 1] 

        # South Face
        if j == 0:
           Sh_coeff[tx, ty, 3] = 0
        else:
            aps = f_y * Sh_ap2d_vel[six, siy] + (1 - f_y) * Sh_ap2d_vel[six, siy - 1]
            ds = dir_area_sx[i, j]**2 + dir_area_sy[i, j]**2
            Sh_coeff[tx, ty, 3] = ds / aps

            if ty == bh - 1:
                aps = f_y_up * Sh_ap2d_vel[six, siy + 1] + (1 - f_y_up) * Sh_ap2d_vel[six, siy]
                ds = dir_area_sx[i, j_up]**2 + dir_area_sy[i, j_up]**2
                Sh_coeff[tx, ty + 1, 3] = ds / aps

        cuda.syncthreads()

        # North
        if j == nj - 1:
            Sh_coeff[tx, ty, 2] = 0 
        else:
            Sh_coeff[tx, ty, 2] = Sh_coeff[tx, ty + 1, 3] 

        # a_p
        coeffs[i, j, 4] = Sh_coeff[tx, ty, 0] + Sh_coeff[tx, ty, 1] + Sh_coeff[tx, ty, 2] + Sh_coeff[tx, ty, 3]

        # Transfer to the return variable
        coeffs[i, j, 0] = Sh_coeff[tx, ty, 0]
        coeffs[i, j, 1] = Sh_coeff[tx, ty, 1]
        coeffs[i, j, 2] = Sh_coeff[tx, ty, 2]
        coeffs[i, j, 3] = Sh_coeff[tx, ty, 3]
    
        # Continuity Error
        su2d[i, j] = conv_W - conv_w[i + 1, j] + conv_S - conv_s[i, j + 1]

         # set coeffs = 0 in [0,0] tp make it non-singular
        if i == 0 and j == 0:
            coeffs[0, 0, 0:4] = 0
            coeffs[0, 0, 4] = 1
        

@cuda.jit
def Correct_Flow_Variables(u2d, v2d, p2d, conv_w, conv_s, coeffsPP, pp2d, dpp_dx, dpp_dy, coeffsUV, vol, ni, nj, urf_p):
   
    # Could be rewritten using shared memory
   
    i, j = cuda.grid(2)
    
    if i < ni and j < nj:
        

        pp = pp2d[i, j]
        volume = vol[i, j]
        ap_vel = coeffsUV[i, j, 4]
        dppdx = dpp_dx[i, j]
        dppdy = dpp_dy[i, j]
            
        # Correct Convections
    
        # West face
        if i > 0: 
            conv_w[i, j] += coeffsPP[i - 1, j, 1] * (pp - pp2d[i - 1, j])

	# South face 
        if j > 0:
            conv_s[i, j] += coeffsPP[i, j - 1, 3] * (pp - pp2d[i, j - 1])

	# Correct p (reference pressure is set at [0, 0] 
        p2d[i, j] += urf_p * (pp - pp2d[0, 0])

	    # Correct Velocities 
        u2d[i, j] -= dppdx * volume / ap_vel
        v2d[i, j] -= dppdy * volume / ap_vel


@cuda.jit
def FindMax(MaxVal, Phi2d):
    
    i,j = cuda.grid(2)
    cuda.atomic.max(MaxVal, (0, 0), Phi2d[i, j])


if __name__ == '__main__':  
    main()

