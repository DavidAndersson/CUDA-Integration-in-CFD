import numpy as np

# Serial implementation of Convection function

def Convection(ap2d_vel, p2d, uface_w, uface_s, vface_w, vface_s, uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types, vBC_e, vBC_w, vBC_n, vBC_s, \
               ni, nj, fx, fy, dir_area_wx, dir_area_wy, dir_area_sx, dir_area_sy, F_area_w, F_area_s):


    # Interpolated a_p coefficient
    ap_w = np.zeros((ni + 1, nj))
    ap_s = np.zeros((ni, nj + 1))

    conv_w = -uface_w * dir_area_wx - vface_w * dir_area_wy
    conv_s = -uface_s * dir_area_sx - vface_s * dir_area_sy

    # West Face

    # create ghost cells at east & west boundaries with Neumann b.c.
    p2d_e = p2d
    p2d_w = p2d

    # duplicate last row and put it at the end
    p2d_e = np.insert(p2d_e, -1, p2d_e[-1, :], axis=0)

    # duplicate row 0 and put it before row 0 (west boundary)
    p2d_w = np.insert(p2d_w, 0, p2d_w[0, :], axis=0)

    dp = np.roll(p2d_e, -1, axis=0) - 3*p2d_e + 3 * \
        p2d_w - np.roll(p2d_w, 1, axis=0)

    ap_w[0:-1, :] = fx * ap2d_vel + (1 - fx) * np.roll(ap2d_vel, 1, axis=0)
    ap_w[-1, :] = 1e-20

    dvelw = dp * F_area_w / 4 / ap_w

    # boundaries (no corrections)
    dvelw[0, :] = 0
    dvelw[-1, :] = 0

    conv_w = conv_w + F_area_w * dvelw

    # South face

    # create ghost cells at north & south boundaries with Neumann b.c.
    p2d_n = p2d
    p2d_s = p2d

    # duplicate last column and put it at the end
    p2d_n = np.insert(p2d_n, -1, p2d_n[:, -1], axis=1)

    # duplicate first column and put it before column 0 (south boundary)
    p2d_s = np.insert(p2d_s, 0, p2d_s[:, 0], axis=1)

    dp = np.roll(p2d_n, -1, axis=1) - 3 * p2d_n + 3 * p2d_s - np.roll(p2d_s, 1, axis=1)

    #  aps[:,1:]=fy*np.roll(ap2d_vel,-1,axis=1)+(1-fy)*ap2d_vel
    ap_s[:, 0:-1] = fy * ap2d_vel + (1 - fy) * np.roll(ap2d_vel, 1, axis=1)
    ap_s[:, -1] = 1e-20

    dvels = dp * F_area_s / 4 / ap_s

    # boundaries (no corrections)
    dvels[:, 0] = 0
    dvels[:, -1] = 0

    conv_s = conv_s + F_area_s * dvels

    # Boundaries

    # East
    if uBC_Types[0] == 'd':
        conv_w[-1, :] = -uBC_e * dir_area_wx[-1, :] - vBC_e * dir_area_wy[-1, :]

    # West
    if uBC_Types[1] == 'd':
        conv_w[0, :] = -uBC_w * dir_area_wx[0, :] - vBC_w * dir_area_wy[0, :]

    # North
    if uBC_Types[2] == 'd':
        conv_s[:, -1] = -uBC_n * dir_area_sx[:, -1] - vBC_n * dir_area_sy[:, -1]

    # South
    if uBC_Types[3] == 'd':
        conv_s[:, 0] = -uBC_s * dir_area_sx[:, 0] - vBC_s * dir_area_sy[:, 0]

    
    return conv_w, conv_s

