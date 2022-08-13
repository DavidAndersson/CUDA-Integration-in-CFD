import numpy as np

def Face_Phi(phi2d, phiBC_e, phiBC_w, phiBC_n, phiBC_s, phiBC_Types, ni, nj, fx, fy):

    phi2d_face_w = np.empty((ni+1, nj))
    phi2d_face_s = np.empty((ni, nj+1))
    
    phi2d_face_w[0:-1, :] = fx * phi2d + (1 - fx) * np.roll(phi2d, 1, axis=0)
    phi2d_face_s[:, 0:-1] = fy * phi2d + (1 - fy) * np.roll(phi2d, 1, axis=1)

    # Handle the boundaries:
    # - If Dirichlet -> apply given values
    # - If Neumann (Homogeneous) -> copy the corresponding value from phi2d

    # East Boundary
    if phiBC_Types[0] == 'd':
        phi2d_face_w[-1, :] = phiBC_e
    else:
        phi2d_face_w[-1, :] = phi2d[-1, :]

    # West Boundary
    if phiBC_Types[1] == 'd':
        phi2d_face_w[0, :] = phiBC_w
    else:
        phi2d_face_w[0, :] = phi2d[0, :]

    # North Boundary
    if phiBC_Types[2] == 'd':
        phi2d_face_s[:, -1] = phiBC_n
    else:
        phi2d_face_s[:, -1] = phi2d[:, -1]

    # South Boundary
    if phiBC_Types[3] == 'd':
        phi2d_face_s[:, 0] = phiBC_s
    else:
        phi2d_face_s[:, 0] = phi2d[:, 0]


    return phi2d_face_w, phi2d_face_s
