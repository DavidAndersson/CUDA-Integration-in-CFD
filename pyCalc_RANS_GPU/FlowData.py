
#### Varaible Flow variables 

# Flow Varialbes at cell centers
u2d = None
v2d = None
p2d = None
k2d = None
pp2d = None
omega2d = None

conv_w = None
conv_s = None

# Face Variables
uface_w = None
vface_w = None
pface_w = None
uface_s = None
vface_s = None
pface_s = None
ppface_w = None
ppface_s = None


# Coefficients
coeffs_UV = None
coeffs_p = None
coeffs_k = None
coeffs_omega = None


#### Constant Flow variables

# Boundary Conditions
uBC_e = None
uBC_w = None
uBC_n = None
uBC_s = None
uBC_Types = None

vBC_e = None
vBC_w = None
vBC_n = None
vBC_s = None
vBC_Types = None

kBC_e = None
kBC_w = None
kBC_n = None
kBC_s = None
kBC_Types = None

omegaBC_e = None
omegaBC_w = None
omegaBC_n = None
omegaBC_s = None
omegaBC_Types = None

# Boundary Coefficients
ae_bndry = None
aw_bndry = None
an_bndry = None
as_bndry = None

# Face Coefficients (interpolated)
ap_w = None
ap_s = None

# 2d viscosity field
vis2d = None
