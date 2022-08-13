import FlowData as FData
import GeometricData as GData
import numpy as np


def Init():

    FData.aw2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ae2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.as2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.an2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.al2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ah2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ap2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.ap2d_vel = np.ones((GData.ni, GData.nj)) * 1e-20

    FData.u2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.v2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.p2d = np.ones((GData.ni, GData.nj)) * 1e-20
    FData.k2d = np.ones((GData.ni, GData.nj)) 
    FData.omega2d = np.ones((GData.ni, GData.nj)) 
   