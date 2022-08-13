import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)

viscos=1/16000


# makes sure figures are updated when using ipython
display.clear_output(wait=True)

datax= np.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

y=yp2d[1,:]
# z grid
zmax, nk=np.loadtxt('z.dat')
nk=np.int(nk)
zp = np.linspace(0, zmax, nk)

itstep,nk,dz=np.load('itstep.npy')
p2d=np.load('p_averaged.npy')/itstep
u2d=np.load('u_averaged.npy')/itstep
v2d=np.load('v_averaged.npy')/itstep
w2d=np.load('w_averaged.npy')/itstep
k2d_model=np.load('k_averaged.npy')/itstep
vis2d=np.load('vis_averaged.npy')/itstep
uu2d=np.load('uu_stress.npy')/itstep
vv2d=np.load('vv_stress.npy')/itstep
ww2d=np.load('ww_stress.npy')/itstep
uv2d=np.load('uv_stress.npy')/itstep
fk2d=np.load('fk_averaged.npy')/itstep
om2d=np.load('om_averaged.npy')/itstep

uu2d=uu2d-u2d**2
uv2d=uv2d-u2d*v2d


u=np.mean(u2d,axis=0)
uu=np.mean(uu2d,axis=0)
vv=np.mean(vv2d,axis=0)
ww=np.mean(ww2d,axis=0)
uv=np.mean(uv2d,axis=0)
k_model=np.mean(k2d_model,axis=0)
fk=np.mean(fk2d,axis=0)
om=np.mean(om2d,axis=0)
vis=np.mean(vis2d,axis=0)

k=0.5*(uu+vv+ww)

vist=vis-viscos
dudy=np.gradient(u,y)
uv_model=-vist*dudy

np.savetxt('y_u_uu_vv_ww_uv_5200_DES_nj96.txt', np.c_[y,u,uu,vv,ww,uv])


ustar=(viscos*u[1]/y[1])**0.5

yplus=y*ustar/viscos



f_e_mean=np.load('f_e_mean.npy')/itstep
f_d_mean=np.load('f_d_mean.npy')/itstep
l_c_mean=np.load('l_c_mean.npy')/itstep
l_tilde_mean=np.load('l_tilde_mean.npy')/itstep
f_dt_mean=np.load('f_dt_mean.npy')/itstep
f_b_mean=np.load('f_b_mean.npy')/itstep
f_e1_mean=np.load('f_e1_mean.npy')/itstep
f_e2_mean=np.load('f_e2_mean.npy')/itstep
denom_mean=np.load('denom_mean.npy')/itstep
f1_sst_mean=np.load('f1_sst_mean.npy')/itstep
f2_sst_mean=np.load('f2_sst_mean.npy')/itstep

r_dt=(vis-1)*viscos/denom_mean
r_dl=viscos/denom_mean

#    denom=kappa**2*dist3d**2*gen**0.5
gen=(denom_mean/0.4**2/y**2)**2

data_sst=np.loadtxt('y-term-1-2-3-f-cross-zeta-c_omega_1-c_omega_2-sst.txt')
#     np.c_[yp2d[1,:],term1[1,:,1],term2[1,:,1],term3[1,:,1],f1_sst[1,:,1],f2_sst[1,:,1,cross_term[1,:,1],zeta[1,:,1],c_sst_1[1,:,1],c_sst_2[1,:,1]])
f1_sst=data_sst[:,4]
f2_sst=data_sst[:,5]
cross_term=data_sst[:,6]
zeta=data_sst[:,7]
c_omega_sst_1=data_sst[:,8]
c_omega_sst_2=data_sst[:,9]

DNS_mean=np.genfromtxt("/chalmers/users/lada//DNS_channel_5200/LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];

DNS_stress=np.genfromtxt("/chalmers/users/lada//DNS_channel_5200/LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2];
v2_DNS=DNS_stress[:,3];
w2_DNS=DNS_stress[:,4];
uv_DNS=DNS_stress[:,5];


k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)


u_reichard=1/0.4*np.log(1+0.4*yplus)+7.8*(1-np.exp(-yplus/11))-yplus*np.exp(-yplus/3);




lst=list(range(len(u_DNS)))
iDNS=lst[0::45]

# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2


########################################## compare vis 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
cmu=0.09
a1=cmu**0.5
denom=np.maximum(a1*om,gen**0.5*f2_sst_mean)
vis_sst= a1*k_model/denom+viscos
vis_k_omega= k_model/om+viscos
  
plt.plot(yplus,vis_sst/viscos,'b-')
plt.plot(yplus,vis_k_omega/viscos,'r--')
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 950, 0, 70])
plt.savefig('compare-vis.eps')





########################################## terms in omega eq
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
c_omega_1= 5./9.
c_omega_2=3./40.
prod_omega=c_omega_1*gen
diss_omega=c_omega_2*om**2
plt.plot(yplus,prod_omega,'b-')
plt.plot(yplus,-diss_omega,'r--')
plt.plot(yplus,cross_term,'k-.')
plt.title("terms in omega eq")
plt.xlabel("$y^+$")
plt.axis([1, 2000, -10, 10])
plt.savefig('prod-destruct-cross-omega-eq.eps')


########################################## f1_sst f2_sst
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus,f1_sst,'b-')
plt.plot(yplus,f2_sst,'r-')
plt.plot(yplus,f1_sst_mean,'b--')
plt.plot(yplus,f2_sst_mean,'r--')
term1b=500*viscos/(om*y**2)

# f2_sst
cmu=0.09
zeta_2=np.maximum(2*k_model**0.5/(cmu*om*y),term1b)
f2_sst_comp=np.tanh(zeta**2)

plt.xlabel("$y^+$")
plt.ylabel("$f_{1,SST} \quad f_{2,SST}$")
plt.axis([1, 2000,0, 1])
plt.savefig('f1_sst-f2_sst.eps',bbox_inches='tight')

########################################## terms vist_sst
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# a1*om3d,gen**0.5*f2_sst
a1=0.09**0.5
term_omega=a1*om
term_gen=gen**0.5*f2_sst_mean
plt.plot(yplus,term_omega-term_gen,'b-')

plt.xlabel("$y^+$")
plt.ylabel("$a_1 \omega \quad |S|f_{2,SST}$")
plt.axis([1, 2000,-200, 200])
plt.savefig('terms-in-visa_sst.eps',bbox_inches='tight')


########################################## vist_sst
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# a1*om3d,gen**0.5*f2_sst
a1=0.09**0.5
term_omega=a1*om
term_gen=gen**0.5*f2_sst_mean
a1=cmu**0.5
denom=np.maximum(a1*om,gen**0.5*f2_sst_mean)
vis_sst= a1*k_model/denom+viscos
vis_calc= k_model/om+viscos

plt.plot(yplus,vis_sst/viscos-1,'b-')
plt.plot(yplus,vis_calc/viscos-1,'r--')
plt.xlabel("$y^+$")
plt.ylabel(r"$\nu_t \quad \nu_{t,SST}$")
plt.axis([1, 2000,0, 120])
plt.savefig('vist-and-vist_sst.eps',bbox_inches='tight')


########################################## U 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.semilogx(yplus,u,'b-')
plt.semilogx(yplus,u_reichard,'bo')
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 33])
plt.savefig('u_log_16000-python.eps',bbox_inches='tight')


########################################## uu 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,uu,'b-')
plt.plot(yplus_DNS[jDNS],u2_DNS[jDNS],'bo')
plt.ylabel(r"$\overline{u'u'}$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 11])
plt.savefig('uu_16000-python.eps',bbox_inches='tight')



########################################## vv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,vv,'b-')
plt.plot(yplus_DNS[iDNS],v2_DNS[iDNS],'bo')
plt.ylabel(r"$\overline{v'v'}$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 2])
plt.savefig('vv_16000-python.eps',bbox_inches='tight')



########################################## ww 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,ww,'b-')
plt.plot(yplus_DNS[iDNS],w2_DNS[iDNS],'bo')
plt.ylabel(r"$\overline{w'w'}$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 2])
plt.savefig('ww_16000-python.eps',bbox_inches='tight')



########################################## uv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,uv_model,'r--')
plt.plot(yplus,uv,'b-')
plt.plot(yplus_DNS[iDNS],uv_DNS[iDNS],'bo')
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, -1, 0])
plt.savefig('uv_16000-python.eps',bbox_inches='tight')


########################################## k 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,k,'b-')
plt.plot(yplus_DNS[iDNS],k_DNS[iDNS],'bo')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 5])
plt.savefig('k_16000-python.eps',bbox_inches='tight')

########################################## k zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,k,'b-')
plt.plot(yplus,k_model,'r--')
plt.plot(yplus_DNS[iDNS],k_DNS[iDNS],'bo')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 1000, 0, 5])
plt.savefig('k_16000-python-zoom.eps',bbox_inches='tight')

########################################## vis 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,vis/viscos,'b-')
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 0, 120])
plt.savefig('vis_16000-python.eps',bbox_inches='tight')


########################################## k 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,fk,'b-')
plt.ylabel(r"$f_k$")
plt.xlabel("$y^+$")
plt.axis([1, 16000, 1, 3.0])
plt.savefig('fk_16000-python.eps',bbox_inches='tight')

########################################## f_d 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(y,f_d_mean,'b-',label='$f_d$')
plt.plot(y,f_b_mean,'r--',label='$f_b$')
plt.plot(y,f_e_mean,'k-.',label='$f_e$')
l_rans=k_model**0.5/om/0.09
l_rat=l_tilde_mean/l_rans
plt.plot(yplus,l_rat,'r-',label=r'$\tilde{l}/l_{RANS}$')
plt.legend(loc="upper right",prop=dict(size=18))
plt.xlabel("$y^+$")
plt.axis([0, 0.2, 0, 1.9])
plt.savefig('f_d_f_b_395-python.eps',bbox_inches='tight')

########################################## f_d  vs yplus
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,f_d_mean,'b-',label='$f_d$')
plt.plot(yplus,f_b_mean,'r--',label='$f_b$')
plt.plot(yplus,f_e_mean,'k-.',label='$f_e$')
l_rans=k_model**0.5/om/0.09
l_rat=l_tilde_mean/l_rans
plt.plot(yplus,l_rat,'r-',label=r'$\tilde{l}/l_{RANS}$')
plt.legend(loc="upper right",prop=dict(size=18))
plt.xlabel("$y^+$")
plt.axis([0, 1000, 0, 1.9])
plt.savefig('f_d_f_b_395-python-zoom.eps',bbox_inches='tight')


########################################## f_d 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(y,1-f_dt_mean,'b-',label='$1-f_d$')
plt.plot(y,f_b_mean,'r--',label='$f_b$')
l_rans=k_model**0.5/om/0.09
l_rat=l_tilde_mean/l_rans
plt.plot(y,l_rat,'r-',label=r'$\tilde{l}/l_{RANS}$')
plt.legend(loc="upper right",prop=dict(size=18))
plt.xlabel("$y^+$")
plt.axis([0, 0.2, 0, 1.2])
plt.savefig('1-f_dt_f_b_395-python.eps',bbox_inches='tight')

