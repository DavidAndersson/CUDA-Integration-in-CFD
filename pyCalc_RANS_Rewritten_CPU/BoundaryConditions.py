import numpy as np
from Constants import viscos
import GeometricData as GData


def GetBC(ni, nj):

    # The boundaries will follow a similar standard as the coefficients. 
    # 0. East, 1. West, 2. North, 3. South 


    uBC_e = np.zeros(nj)
    uBC_w = np.zeros(nj)
    uBC_n = np.ones(ni) # Note 
    uBC_s = np.zeros(ni)
    uBC_Types = np.char.array(['d', 'd', 'd', 'd'])
    

    vBC_e = np.zeros(nj)
    vBC_w = np.zeros(nj)
    vBC_n = np.zeros(ni)
    vBC_s = np.zeros(ni)
    vBC_Types = np.char.array(['d', 'd', 'd', 'd'])


    pBC_e = np.zeros(nj)
    pBC_w = np.zeros(nj)
    pBC_n = np.zeros(ni)
    pBC_s = np.zeros(ni)
    pBC_Types = np.char.array(['n', 'n', 'n', 'n'])


    kBC_e = np.zeros(nj)
    kBC_w = np.ones(nj)*1e-2
    kBC_w[10:] = 1e-5
    kBC_n = np.ones(ni)*1e-5
    kBC_s = np.zeros(ni)
    kBC_Types = np.char.array(['d', 'd', 'n', 'd'])


    omegaBC_e = np.zeros(nj)
    omegaBC_w = np.ones(nj)
    omegaBC_n = np.zeros(ni)
    omegaBC_s = np.zeros(ni)
    omegaBC_Types = np.char.array(['n', 'd', 'n', 'd'])

    xwall_s = 0.5*(GData.x2d[0:-1, 0] + GData.x2d[1:, 0])
    ywall_s = 0.5*(GData.y2d[0:-1, 0] + GData.y2d[1:, 0])
    dist2_s = (GData.yp2d[:, 0] - ywall_s)**2 + (GData.xp2d[:, 0]-xwall_s)**2

    omegaBC_s = 10 * 6 * viscos / 0.075 / dist2_s

    return ( uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types,
             vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types,
             pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types,
             kBC_e, kBC_w, kBC_n, kBC_s, kBC_Types,
             omegaBC_e, omegaBC_w, omegaBC_n, omegaBC_s, omegaBC_Types ) 
