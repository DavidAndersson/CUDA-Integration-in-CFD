import numpy as np

ni=2048
nj=2048

yfac=1.05 # stretching

if yfac == 1:
    xFile = "x_" + str(ni) + "x" + str(nj) + "_Equi" + ".dat"
    yFile = "y_" + str(ni) + "x" + str(nj) + "_Equi" + ".dat"
    
elif yfac > 1:
    xFile = "x_" + str(ni) + "x" + str(nj) + ".dat"
    yFile = "y_" + str(ni) + "x" + str(nj) + ".dat"


ymax=2
xmax=3
viscos=1/1000
dy=0.1
yc=np.zeros(nj+1)
yc[0]=0.
for j in range(1,int(nj/2)+1):
    yc[j]=yc[j-1]+dy
    dy=yfac*dy


ymax_scale=yc[int(nj/2)]

# cell faces
for j in range(1,int(nj/2)+1):
   yc[j]=yc[j]/ymax_scale
   yc[nj-j+1]=ymax-yc[j-1]

yc[int(nj/2)]=1


print('y+',0.5*yc[1]/viscos)
# make it 2D
y2d=np.repeat(yc[None,:], repeats=ni+1, axis=0)

y2d=np.append(y2d,nj)
np.savetxt(yFile, y2d)

# x grid
xc=yc
# make it 2D
x2d=np.repeat(xc[:,None], repeats=nj+1, axis=1)
x2d_org=x2d
x2d=np.append(x2d,ni)
np.savetxt(xFile, x2d)




# check it
datay= np.loadtxt(xFile)
y=datay[0:-1]
nj=int(datay[-1])

y2=np.zeros((ni+1,nj+1))
y2=np.reshape(y,(ni+1,nj+1))

datax= np.loadtxt(yFile)
x=datax[0:-1]
ni=int(datax[-1])

x2=np.zeros((ni+1,nj+1))
x2=np.reshape(x,(ni+1,nj+1))



