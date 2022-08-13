import numpy as np
import os.path

##### NOTE!!! ################
# This Grid function does not work for domains at 2048x2048 or larger. 

def Generate_Mesh(ni, nj):
        

    xFile = "x_" + str(ni) + "x" + str(nj) + ".dat"
    yFile = "y_" + str(ni) + "x" + str(nj) + ".dat"
    
    if os.path.exists(xFile) and os.path.exists(yFile):
        return xFile, yFile
    
    
    print("Generating Mesh ...")
    
    yfac = 1.05  # stretching
    
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


    print("Finished Generating Mesh!")

    return xFile, yFile



