import scipy.io as sio
import sys
import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)

viscos=3.57E-5

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

u2d=np.load('u2d_saved.npy')
p2d=np.load('p2d_saved.npy')
v2d=np.load('v2d_saved.npy')



########################################## iso u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,u2d, cmap=plt.get_cmap('hot'),shading='gouraud')
#plt.pcolor(xp2d,yp2d,u2d, cmap=plt.get_cmap('hot'),shading='gouraud')
#plt.pcolor(xp2d,yp2d,vis2d)
#colormap = plt.get_cmap('hot')
#plt.pcolormesh(xp2d,yp2d,omega_3, vmin=-1,vmax=1,cmap=plt.get_cmap('hot'),shading='gouraud')
p.arrow( 0.7, 2.1, 0.6, 0, fc="k", ec="k",head_width=0.05, head_length=0.1 )
plt.text(1,2.2,'$U_{wall}$')
plt.axis('equal')
#plt.colorbar()
plt.axis('off')
plt.box(on=None)
plt.savefig('u_iso.png')


########################################## grid
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# plt.pcolormesh(xp2d,yp2d,v3d[:,:,15], cmap=plt.get_cmap('hot'),shading='gouraud')

#%%%%%%%%%%%%%%%%%%%%% grid
for i in range(0,ni+1):
   plt.plot(x2d[i,:],y2d[i,:],'k-')

for j in range(0,nj+1):
   plt.plot(x2d[:,j],y2d[:,j],'k-')

plt.axis('equal')
plt.axis('off')
p.arrow( 0.7, 2.1, 0.6, 0, fc="k", ec="k",head_width=0.05, head_length=0.1 )
plt.text(1,2.2,'$U_{wall}$')
plt.box(on=None)
plt.savefig('grid.png')


