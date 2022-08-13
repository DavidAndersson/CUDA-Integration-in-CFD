
def setup_case():
   import numpy as np
   import sys


########### section 1 choice of differencing scheme ###########
   scheme = 'h'  #hybrid
   scheme_turb = 'h'  #hybrid upwind-central 

########### section 2 turbulence models ###########
   cmu = 0.09
   kom  =  False
   c_omega_1 =  5./9.
   c_omega_2 = 3./40.
   prand_omega = 2.0
   prand_k = 2.0

########### section 3 restart/save ###########
   restart  =  False
   save  =  True

########### section 4 fluid properties ###########
   viscos = 1/1000

########### section 5 relaxation factors ###########
   urfvis = 0.5
   urf_vel = 0.5
   urf_k = 0.5
   urf_p = 1.0
   urf_omega = 0.5

########### section 6 number of iteration and convergence criterira ###########
   maxit = 200
   min_iter = 1
   sormax = 1e-20

   solver_vel = 'lgmres'
   solver_pp = 'lgmres'
   solver_vel = 'direct'
   solver_pp = 'direct'
   solver_turb = 'gmres'
   nsweep_vel = 50
   nsweep_pp = 50
   nsweep_kom = 1
   convergence_limit_vel = 1e-6
   convergence_limit_u = 1e-6
   convergence_limit_v = 1e-6
   convergence_limit_w = 1e-6
   convergence_limit_k = 1e-10
   convergence_limit_om = 1e-10
   convergence_limit_pp = 5e-4

########### section 7 all variables are printed during the iteration at node ###########
   imon = ni-10
   jmon = int(nj/2)

########### section 8 save data for post-processing ###########
   vtk = False
   save_all_files = False
   vtk_file_name = 'bound'

########### section 9 residual scaling parameters ###########
   uin = 1
   resnorm_p = uin*y2d[1,-1]
   resnorm_vel = uin**2*y2d[1,-1]


########### Section 10 boundary conditions ###########

# boundary conditions for u
   u_bc_west = np.zeros(nj)
   u_bc_east = np.zeros(nj)
   u_bc_south = np.zeros(ni)
   u_bc_north = np.ones(ni)

   u_bc_west_type = 'd' 
   u_bc_east_type = 'd' 
   u_bc_south_type = 'd'
   u_bc_north_type = 'd'

# boundary conditions for v
   v_bc_west = np.zeros(nj)
   v_bc_east = np.zeros(nj)
   v_bc_south = np.zeros(ni)
   v_bc_north = np.zeros(ni)

   v_bc_west_type = 'd' 
   v_bc_east_type = 'd' 
   v_bc_south_type = 'd'
   v_bc_north_type = 'd'

# boundary conditions for p
   p_bc_west = np.zeros(nj)
   p_bc_east = np.zeros(nj)
   p_bc_south = np.zeros(ni)
   p_bc_north = np.zeros(ni)

   p_bc_west_type = 'n'
   p_bc_east_type = 'n'
   p_bc_south_type = 'n'
   p_bc_north_type = 'n'

# boundary conditions for k
   k_bc_west = np.ones(nj)*1e-2
   k_bc_west[10:] = 1e-5
   k_bc_east = np.zeros(nj)
   k_bc_south = np.zeros(ni)
   k_bc_north = np.ones(ni)*1e-5

   k_bc_west_type = 'd'
   k_bc_east_type = 'd'
   k_bc_south_type = 'd'
   k_bc_north_type = 'n'

# boundary conditions for omega
   om_bc_west = np.ones(nj)
   om_bc_east = np.zeros(nj)
   om_bc_south = np.zeros(ni)
   om_bc_north = np.zeros(ni)

   xwall_s = 0.5*(x2d[0:-1,0]+x2d[1:,0])
   ywall_s = 0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist2_s = (yp2d[:,0]-ywall_s)**2+(xp2d[:,0]-xwall_s)**2
   om_bc_south = 10*6*viscos/0.075/dist2_s

   om_bc_west_type = 'd'
   om_bc_east_type = 'n'
   om_bc_south_type = 'd'
   om_bc_north_type = 'n'

   return 


