import numpy as np
from scipy import sparse
from scipy.sparse import spdiags, linalg, eye

def Solve_2d(phi2d, coeffs, su2d, tol_conv, nmax, solver_local, ni, nj):

   a_e = np.matrix.flatten(coeffs[:,:, 0]) 
   a_w = np.matrix.flatten(coeffs[:,:, 1])
   a_n = np.matrix.flatten(coeffs[:,:, 2])
   a_s = np.matrix.flatten(coeffs[:,:, 3])
   a_p = np.matrix.flatten(coeffs[:,:, 4])

   s_u = np.matrix.flatten(su2d)
   phi = np.matrix.flatten(phi2d)

   A = sparse.diags([a_p, -a_n[0:-1], -a_s[1:], -a_e, -a_w[nj:]], [0, 1, -1, nj, -nj], format='csr')

   res_su = np.linalg.norm(s_u)
   resid_init = np.linalg.norm(A * phi - s_u)

   phi_org = phi

# bicg (BIConjugate Gradient)
# bicgstab (BIConjugate Gradient STABilized)
# cg (Conjugate Gradient) - symmetric positive definite matrices only
# cgs (Conjugate Gradient Squared)
# gmres (Generalized Minimal RESidual)
# minres (MINimum RESidual)
# qmr (Quasi
   if solver_local == 'direct':
      info = 0
      resid = np.linalg.norm(A*phi - s_u)
      phi = linalg.spsolve(A, s_u)
#    if solver_local == 'pyamg':
#       if iter == 0:
#          print('solver in solve_2d: pyamg solver')
#       App = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
    #   res_amg = []
    #   phi = App.solve(su, tol=tol_conv, x0=phi, residuals=res_amg)
    #   info=0
    #   print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])
    
   elif solver_local == 'cgs':
      phi,info=linalg.cgs(A,s_u,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
      
   elif solver_local == 'cg':
      phi,info=linalg.cg(A,s_u,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
      
   elif solver_local == 'gmres':
      phi,info=linalg.gmres(A,s_u,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
      
   elif solver_local == 'qmr':
      phi,info=linalg.qmr(A,s_u,x0=phi, tol=tol_conv, atol=tol_conv,  maxiter=nmax)  # good
      
   elif solver_local == 'lgmres':
      phi, info = linalg.lgmres(A, s_u, x0=phi, tol=tol_conv, atol=tol_conv, maxiter=nmax)  # good
   
   #if info > 0:
     #print('warning in module solve_2d: convergence in sparse matrix solver not reached')


   # compute residual without normalizing with |b|=|su2d|
   if solver_local != 'direct':
      resid = np.linalg.norm(A*phi - s_u)

   delta_phi = np.max(np.abs(phi-phi_org))

   phi2d = np.reshape(phi, (ni, nj))
   phi2d_org = np.reshape(phi_org, (ni, nj))

   #if solver_local != 'pyamg':
      #print(f"{'residual history in solve_2d: initial residual: '} {resid_init:.2e}{'final residual: ':>30}{resid:.2e}\
      #{'delta_phi: ':>25}{delta_phi:.2e}")

# we return the initial residual; otherwise the solution is always satisfied (but the non-linearity is not accounted for)
   return phi2d, resid_init
