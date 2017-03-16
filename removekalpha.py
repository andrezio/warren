#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     14/03/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

namefile = 'ZnO_DEF_101_F_XY.xy'


x,y = np.loadtxt(namefile, unpack= True)


y1=[]
for i in range(len(y)):
    try:
        y1.append(y[i]-y[i+1]/3)
    except:
        y1.append(y[i])


plt.plot(x,y,linestyle='-', marker='o')

plt.plot(x,y1,linestyle='-', marker='o')

plt.show()