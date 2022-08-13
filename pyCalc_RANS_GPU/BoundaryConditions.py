import numpy as np

def GetBC(ni, nj):

    # Since characters do not work in CUDA this mapping needs to be done to the boundary types:
    # 1. Dirichlet
    # 2. Homogeneous Neumann

    uBC_e = np.zeros(nj)
    uBC_w = np.zeros(nj)
    uBC_n = np.ones(ni) # Note ni
    uBC_s = np.zeros(ni)
    uBC_Types = np.array([1, 1, 1, 1])
    

    vBC_e = np.zeros(nj)
    vBC_w = np.zeros(nj)
    vBC_n = np.zeros(ni)
    vBC_s = np.zeros(ni)
    vBC_Types = np.array([1, 1, 1, 1])


    pBC_e = np.zeros(nj)
    pBC_w = np.zeros(nj)
    pBC_n = np.zeros(ni)
    pBC_s = np.zeros(ni)
    pBC_Types = np.array([2, 2, 2, 2])

    return ( uBC_e, uBC_w, uBC_n, uBC_s, uBC_Types,
             vBC_e, vBC_w, vBC_n, vBC_s, vBC_Types,
             pBC_e, pBC_w, pBC_n, pBC_s, pBC_Types) 
