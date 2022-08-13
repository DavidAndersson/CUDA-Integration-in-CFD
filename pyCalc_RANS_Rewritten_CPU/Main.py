
from Imports import *
ProgStart = time.time()

# User Input
DomainSizeX = 512
DomainSizeY = 512

xFile, yFile = Generate_Mesh(DomainSizeX, DomainSizeY)

Setup(xFile, yFile)

# Import everything from GeometricData, FlowData and Constants after Setup() has initialized all the values
from GeometricData import *
from FlowData import *
from Constants import *

iterStart  = time.time() 
TotalIterTime = 0

for iter in range(maxit):

    iterTime = time.time()
     
    # Get coefficients for u and v (except for a_p)
    coeffs_UV = Coeff(coeffs_UV, conv_w, conv_s, vis2d, 1, scheme, ni, nj, fx, fy, F_area_w, F_area_s, vol, viscos)
    
    # u2d
    su2d, sp2d = Sources_From_BC(uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos) 
    dp_dx = dphi_dx(pface_w, pface_s, dir_area_wx, dir_area_sx, vol)
    su2d, sp2d, coeffs_UV[:,:,4] = Coeffs_And_Sources_For_U(su2d, sp2d, coeffs_UV, conv_w, u2d, uBC_w, vis2d, dp_dx, aw_bndry, vol, viscos, urf_vel)
    u2d, res.u = Solve_2d(u2d, coeffs_UV, su2d, convergence_limit_u, nsweep_vel, solver_vel, ni, nj)

    
    # v2d
    su2d, sp2d = Sources_From_BC(vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos)
    dp_dy = dphi_dy(pface_w, pface_s, dir_area_wy, dir_area_sy, vol)
    su2d, sp2d, coeffs_UV[:,:,4] = Coeffs_And_Sources_For_V(su2d, sp2d, coeffs_UV, conv_w, v2d, vBC_w, vis2d, dp_dy, aw_bndry, vol, viscos, urf_vel)
    v2d, res.v = Solve_2d(v2d, coeffs_UV, su2d, convergence_limit_v, nsweep_vel, solver_vel, ni, nj)

    
    # pp2d
    uface_w, uface_s = Face_Phi(u2d, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ni, nj, fx, fy)
    vface_w, vface_s = Face_Phi(v2d, vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ni, nj, fx, fy)

    conv_w, conv_s = Convection(coeffs_UV[:,:,4], u2d, v2d, p2d, uface_w, uface_s, vface_w, vface_s, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types,\
                                vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ni, nj, fx, fy, dir_area_wx, \
                                dir_area_wy,  dir_area_sx,  dir_area_sy, F_area_w, F_area_s)
                                

    conv_w = modify_outlet(conv_w, F_area_w)
    coeffs_pp, su2d = Coeffs_And_Sources_For_P(pp2d, coeffs_UV[:,:,4], conv_w, conv_s, ni, nj, fx, fy, \
                                               dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, urf_vel)
                                                                     
    pp2d = np.zeros((ni, nj))
    pp2d, res.p = Solve_2d(pp2d, coeffs_pp, su2d, convergence_limit_pp, nsweep_pp, solver_pp, ni, nj)


    # Correct u, v, p
    ppface_w, ppface_s = Face_Phi(pp2d, pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types, ni, nj, fx, fy)

    dpp_dx = dphi_dx(ppface_w, ppface_s, dir_area_wx, dir_area_sx, vol)
    dpp_dy = dphi_dy(ppface_w, ppface_s, dir_area_wy, dir_area_sy, vol)

    u2d, v2d, p2d, conv_w, conv_s = Correct_Flow_Variables(u2d, v2d, p2d, conv_w, conv_s, coeffs_pp[:,:,1], coeffs_pp[:,:,3], pp2d,\
                                                           dpp_dx, dpp_dy, coeffs_UV[:,:,4], vol)
                                                           
    conv_w = modify_outlet(conv_w, F_area_w)

    
    # Continuity Error
    su2d = conv_w[0:-1,:] - np.roll(conv_w[0:-1,:], -1, axis=0) + conv_s[:,0:-1] - np.roll(conv_s[:,0:-1], -1, axis=1)
    res.pp = abs(np.sum(su2d))
    
    uface_w, uface_s = Face_Phi(u2d, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, ni, nj, fx, fy)
    vface_w, vface_s = Face_Phi(v2d, vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types, ni, nj, fx, fy)
    pface_w, pface_s = Face_Phi(p2d, pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types, ni, nj, fx, fy)

    
    if k_omega:
        
        # k
        dudx = dphi_dx(uface_w, uface_s, dir_area_wx, dir_area_sx, vol)
        dvdx = dphi_dx(vface_w, vface_s, dir_area_wx, dir_area_sx, vol)
        dudy = dphi_dy(uface_w, uface_s, dir_area_wy, dir_area_sy, vol)
        dvdy = dphi_dy(vface_w, vface_s, dir_area_wy, dir_area_sy, vol)

        coeffs_k = Coeff(coeffs_k, conv_w, conv_s, vis2d, prand_k, scheme_turb, ni, nj, fx, fy, F_area_w, F_area_s, vol, viscos)
        su2d, sp2d = Sources_From_BC(kBC_e, kBC_w, kBC_n, kBC_s, kBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos)
        su2d, sp2d, gen, coeffs_k[:,:, 4] = Coeffs_And_Sources_For_k(k2d, kBC_w, aw_bndry, omega2d, conv_w, dudx, dudy, dvdx, dvdy,\
                                                                     su2d, sp2d, coeffs_k, vis2d, vol, viscos, cmu, urf_k)
    
    
        # omega
        coeffs_omega = Coeff(coeffs_omega, conv_w, conv_s, vis2d, prand_omega, scheme_turb, ni, nj, fx, fy, F_area_w, F_area_s, vol, viscos)
        su2d, sp2d = Sources_From_BC(omegaBC_e, omegaBC_w, omegaBC_n, omegaBC_s, omegaBC_Types, ae_bndry, aw_bndry, an_bndry, as_bndry, ni, nj, viscos) 
        su2d, sp2d, coeffs_omega[:,:,4] = Coeffs_And_Sources_Omega(omega2d, omegaBC_w, aw_bndry, conv_w, su2d, sp2d, coeffs_omega, vis2d, vol, gen, urf_omega, viscos, c_omega_1, c_omega_2, urf_vel)
        mega2d, res.om = Solve_2d(omega2d, coeffs_omega , su2d, convergence_limit_om, nsweep_kom, solver_turb, ni, nj)
    
    

    # scale residuals
    res.u /= resnorm_vel
    res.v /= resnorm_vel
    res.p /= resnorm_p

    resmax = np.max([res.u, res.v, res.p])
    
    
    print("________________________________________________________________")
    print("\nIteration: " + str(iter) + "\t Iteration Time: " + str(round(time.time() - iterTime, 8)) + 's')
    print("\nResiduals: ")
    print("max: " + str("{:e}".format(resmax)) + 
          "\t u: " + str("{:e}".format(res.u)) + 
          "\t v: " + str("{:e}".format(res.v)) + 
          "\t pp: " + str("{:e}".format(res.p)))


    umax = np.max(u2d.flatten())
    vmax = np.max(v2d.flatten())
    
    print("\nData: ")
    print("U max: " + "{:e}".format(umax) + 
          "\t V max: " + "{:e}".format(vmax))


    TotalIterTime += time.time() - iterTime

    if resmax < sormax: 
       break
       

print("\n\niteration duration: " + str(time.time() - iterStart) + "s")
print("Average iteration duration: " + str(TotalIterTime / maxit) + "s")
print("Prgram duration: " + str(time.time() - ProgStart) + "s")




