
from numba import cuda
from GeometricData import *
from FlowData import *
from Constants import *
import numpy as np

# Copies over, and creates, all the variables that are used in the solution process

#               GEOMETRIC DATA

dir_area_wx = cuda.to_device(np.float32(dir_area_wx))
dir_area_wy = cuda.to_device(np.float32(dir_area_wy))
dir_area_sx = cuda.to_device(np.float32(dir_area_sx))
dir_area_sy = cuda.to_device(np.float32(dir_area_sy))

F_area_w = cuda.to_device(np.float32(F_area_w))
F_area_s = cuda.to_device(np.float32(F_area_s))

vol = cuda.to_device(np.float32(vol))

fx = cuda.to_device(np.float32(fx))
fy = cuda.to_device(np.float32(fy))

###########################

#               FLOW DATA

u2d = cuda.to_device(np.float32(u2d))
v2d = cuda.to_device(np.float32(v2d))
p2d = cuda.to_device(np.float32(p2d))
pp2d = cuda.to_device(np.float32(pp2d))

conv_w = cuda.to_device(np.float32(conv_w))
conv_s = cuda.to_device(np.float32(conv_s))

uface_w = cuda.to_device(np.float32(uface_w))
vface_w = cuda.to_device(np.float32(vface_w))
pface_w = cuda.to_device(np.float32(pface_w))
ppface_w = cuda.to_device(np.float32(np.zeros((ni + 1, nj))))
uface_s = cuda.to_device(np.float32(uface_s))
vface_s = cuda.to_device(np.float32(vface_s))
pface_s = cuda.to_device(np.float32(pface_s))
ppface_s = cuda.to_device(np.float32(np.zeros((ni, nj + 1))))

# Correction variables for convection
velCorr_w = cuda.to_device(np.float32(np.zeros((ni, nj))))
velCorr_s = cuda.to_device(np.float32(np.zeros((ni, nj))))


# Coefficients
coeffs_UV = cuda.to_device(np.float32(coeffs_UV))
coeffs_pp = cuda.to_device(np.float32(coeffs_pp))

su2d = cuda.to_device(np.float32(np.zeros((ni, nj))))
sp2d = cuda.to_device(np.float32(np.zeros((ni, nj))))

#### Constant Flow variables

# Boundary Conditions
uBC_e = cuda.to_device(np.float32(uBC_e))
uBC_w = cuda.to_device(np.float32(uBC_w))
uBC_n = cuda.to_device(np.float32(uBC_n))
uBC_s = cuda.to_device(np.float32(uBC_s))


vBC_e = cuda.to_device(np.float32(vBC_e))
vBC_w = cuda.to_device(np.float32(vBC_w))
vBC_n = cuda.to_device(np.float32(vBC_n))
vBC_s = cuda.to_device(np.float32(vBC_s))


# Boundary Coefficients
ae_bndry = cuda.to_device(np.float32(ae_bndry))
aw_bndry = cuda.to_device(np.float32(aw_bndry))
an_bndry = cuda.to_device(np.float32(an_bndry))
as_bndry = cuda.to_device(np.float32(as_bndry))

# Face Coefficients (interpolated)
#ap_w = cuda.to_device(np.float32(ap_w))
#ap_s = cuda.to_device(np.float32(ap_s))

# 2d viscosity field
vis2d = cuda.to_device(np.float32(vis2d))


dp_dx = cuda.to_device(np.float32(np.zeros((ni, nj))))
dp_dy = cuda.to_device(np.float32(np.zeros((ni, nj))))
dpp_dx = cuda.to_device(np.float32(np.zeros((ni, nj))))
dpp_dy = cuda.to_device(np.float32(np.zeros((ni, nj))))






