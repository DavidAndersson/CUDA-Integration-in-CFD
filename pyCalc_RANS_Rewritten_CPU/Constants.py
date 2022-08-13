# Global Constants

# Choice of differencing scheme
# Schemes: 1. Hybrid
#          2. c ? upwind central?

scheme = 1        
scheme_turb = 1   # hybrid upwind-central 


# Turbulence Models
cmu = 0.09
k_omega  =  False
c_omega_1 =  5/9
c_omega_2 = 3/40
prand_omega = 2
prand_k = 2

# Restart / Save
restart  =  False
save  =  True

# Fluid properties 
viscos = 1/1000

# Relaxation factors 
urfvis = 0.5
urf_vel = 0.5
urf_k = 0.5
urf_p = 1.0
urf_omega = 0.5

# Number of iteration and convergence criterira
maxit = 200
min_iter = 1
sormax = 1e-20

solver_vel = 'gmres'
solver_pp = 'gmres'
solver_turb = 'gmres'
nsweep_vel = 40
nsweep_pp = 40
nsweep_kom = 1
convergence_limit_vel = 1e-6
convergence_limit_u = 1e-6
convergence_limit_v = 1e-6
convergence_limit_w = 1e-6
convergence_limit_k = 1e-10
convergence_limit_om = 1e-10
convergence_limit_pp = 5e-4

# Set monitors at some node -> Set the value at Setup.py
i_monitor = None
j_monitor = None

# Save data for post-processing
vtk = False
save_all_files = False
vtk_file_name = 'bound'

# Residual scaling parameters
uin = 1
resnorm_p = None
resnorm_vel = None



