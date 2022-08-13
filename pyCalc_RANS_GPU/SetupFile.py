import numpy as np
import GeometricData as GData
import FlowData as FData
import Constants as consts
from BoundaryConditions import GetBC
from Face_Phi import Face_Phi
from Convection import Convection


def Setup(fileX, fileY):

    #############     GEOMETRIC DATA     ##################

    # Read Coordinate data:
    data_x = np.loadtxt(fileX)
    x = data_x[0:-1]
    ni = int(data_x[-1])

    data_y = np.loadtxt(fileY)
    y = data_y[0:-1]
    nj = int(data_y[-1])

    x2d = np.zeros((ni+1, nj+1))
    y2d = np.zeros((ni+1, nj+1))

    x2d = np.reshape(x, (ni+1, nj+1))
    y2d = np.reshape(y, (ni+1, nj+1))

    xp2d = 0.25 * (x2d[0:-1, 0:-1] + x2d[0:-1, 1:] +
                   x2d[1:, 0:-1] + x2d[1:, 1:])
    yp2d = 0.25 * (y2d[0:-1, 0:-1] + y2d[0:-1, 1:] +
                   y2d[1:, 0:-1] + y2d[1:, 1:])

    GData.ni = ni
    GData.nj = nj

    #  west face coordinate
    xw = 0.5*(x2d[0: -1, 0: -1] + x2d[0: -1, 1:])
    yw = 0.5*(y2d[0: -1, 0: -1] + y2d[0: -1, 1:])

    # Interpolation factors (x)
    del1x = ((xw - xp2d)**2 + (yw - yp2d)**2)**0.5
    del2x = ((xw - np.roll(xp2d, 1, axis=0))**2 +
             (yw - np.roll(yp2d, 1, axis=0))**2)**0.5
    GData.fx = del2x / (del1x + del2x)

    #  south face coordinate
    xs = 0.5*(x2d[0: -1, 0: -1] + x2d[1:, 0: -1])
    ys = 0.5*(y2d[0: -1, 0: -1] + y2d[1:, 0: -1])

    # Interpolation factors (y)
    del1y = ((xs - xp2d)**2 + (ys - yp2d)**2)**0.5
    del2y = ((xs - np.roll(xp2d, 1, axis=1))**2 +
             (ys - np.roll(yp2d, 1, axis=1))**2)**0.5
    GData.fy = del2y / (del1y + del2y)


    # Directional Face Areas
    GData.dir_area_wy = np.diff(x2d, axis=1)
    GData.dir_area_wx = -np.diff(y2d, axis=1)
    GData.dir_area_sy = -np.diff(x2d, axis=0)
    GData.dir_area_sx = np.diff(y2d, axis=0)

    # Real Face Areas
    GData.F_area_w = (GData.dir_area_wx**2 + GData.dir_area_wy**2)**0.5
    GData.F_area_s = (GData.dir_area_sx**2 + GData.dir_area_sy**2)**0.5

    # Volume approaximated as the vector product of two triangles for cells
    ax = np.diff(x2d, axis=1)
    ay = np.diff(y2d, axis=1)
    bx = np.diff(x2d, axis=0)
    by = np.diff(y2d, axis=0)

    areaz_1 = 0.5*np.absolute(ax[0: -1, :] * by[:, 0: -1] - ay[0: -1, :]*bx[:, 0: -1])

    ax = np.diff(x2d, axis=1)
    ay = np.diff(y2d, axis=1)
    areaz_2 = 0.5*np.absolute(ax[1:, :]*by[:, 0: -1] - ay[1:, :]*bx[:, 0: -1])

    GData.vol = areaz_1 + areaz_2

    #############     FLOW DATA     ##################

    # Initialize some variables
    FData.aw2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ae2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.as2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.an2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ap2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ap2d_vel = np.ones((GData.ni, GData.nj)) * 1e-20

    FData.u2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.v2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.p2d = np.ones((GData.ni, GData.nj)) * 1e-20

    
    # Read in boundary conditions
    (FData.uBC_e, FData.uBC_w, FData.uBC_n, FData.uBC_s, FData.uBC_Types,
    FData.vBC_e, FData.vBC_w, FData.vBC_n, FData.vBC_s,FData. vBC_Types,
    FData.pBC_e, FData.pBC_w, FData. pBC_n, FData.pBC_s, FData.pBC_Types) = GetBC(GData.ni, GData.nj)

    # Get face values of the flow variables
    FData.uface_w, FData.uface_s = Face_Phi(FData.u2d, FData.uBC_e, FData.uBC_w, FData.uBC_n, FData.uBC_s, FData.uBC_Types, GData.ni, GData.nj, GData.fx, GData.fy)
    FData.vface_w, FData.vface_s = Face_Phi(FData.v2d, FData.vBC_e, FData.vBC_w, FData.vBC_n, FData.vBC_s, FData.vBC_Types, GData.ni, GData.nj, GData.fx, GData.fy)
    FData.pface_w, FData.pface_s = Face_Phi(FData.p2d, FData.pBC_e, FData.pBC_w, FData.pBC_n, FData.pBC_s, FData.pBC_Types, GData.ni, GData.nj, GData.fx, GData.fy)

    # Coeff at the walls (without viscosity)

    FData.ae_bndry = GData.F_area_w[-1, :]**2 / (0.5*GData.vol[-1, :])
    FData.aw_bndry = GData.F_area_w[0, :]**2 / (0.5*GData.vol[0, :])
    FData.an_bndry = GData.F_area_s[:,  -1]**2 / (0.5*GData.vol[:,  -1])
    FData.as_bndry = GData.F_area_s[:, 0]**2 / (0.5*GData.vol[:, 0])

    # Compute initial convections
    #ap2d_vel = np.ones((GData.ni, GData.nj))*1e-20

    FData.conv_w, FData.conv_s = Convection(FData.ap2d_vel, FData.p2d, FData.uface_w,FData. uface_s, FData.vface_w, FData.vface_s, FData.uBC_e, FData.uBC_w, FData.uBC_n, FData.uBC_s, FData.uBC_Types,\
                                            FData.vBC_e, FData.vBC_w, FData.vBC_n, FData.vBC_s, \
                                            GData.ni, GData.nj, GData.fx, GData.fy, GData.dir_area_wx,  GData.dir_area_wy,  GData.dir_area_sx,  GData.dir_area_sy, \
                                            GData.F_area_w, GData.F_area_s)


    # Initialize viscosity field and pressure correction
    FData.vis2d = np.ones((GData.ni, GData.nj)) * consts.viscos
    FData.pp2d = np.zeros((GData.ni, GData.nj))


    # Follows the convention: 0. East, 1. West, 2. North, 3. South, and the last is 4. a_p
    FData.coeffs_UV = np.zeros((ni, nj, 5))
    FData.coeffs_pp = np.zeros((ni, nj, 5))
    FData.coeffs_k = np.zeros((ni, nj, 5))
    FData.coeffs_omega = np.zeros((ni, nj, 5))


   #############     CONSTANT DATA     ##################

    consts.resnorm_p = consts.uin * y2d[1, -1]
    consts.resnorm_vel = consts.uin**2 * y2d[1, -1]


    