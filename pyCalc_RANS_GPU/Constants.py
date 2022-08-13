
# Global Constants

# Choice of differencing scheme

# Schemes: 1. Hybrid
#          2. c ? upwind central?

scheme = 1        
scheme_turb = 1   # hybrid upwind-central 

# Fluid properties 
viscos = (1/1000)

# Relaxation factors 
urfvis = (0.5)
urf_vel = (0.5)
urf_p = (1.0)

# Number of iteration and convergence criterira
maxit = 200
sormax = 1e-20

solver_vel = 'gmres'
solver_pp = 'gmres'
nsweep_vel = 50
nsweep_pp = 50

convergence_limit_u = 1e-6
convergence_limit_v = 1e-6
convergence_limit_pp = 5e-4


# Residual scaling parameters
uin = 1
resnorm_p = None
resnorm_vel = None
