#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lgallego
#
# Created:     21/02/2017
# Copyright:   (c) lgallego 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from Tkinter import *
from ttk import *
import sys
import tkMessageBox
from tkFileDialog   import askopenfilename
import matplotlib.pyplot as plt
from math import sin, cos
import numpy as np
from tkFileDialog   import askopenfilename
import copy
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel
from math import sin,cos,pi,radians,tan,sqrt,log1p
from scipy import stats

namefile="ZnO_DEF_101_F_b_ka2.asc"

x,y = np.loadtxt(namefile, unpack= True)


newy=[]
for i in y:
    newy.append(i)
y=newy

xx=[]
yy=[]



newy=[]
newx=[]
for i in range(len(y)):
    if i == y.index(max(y))*2:
        break
    else:
        newx.append(x[i])
        newy.append(y[i])

x=newx
y=newy

for i in x:
    xx.append(x[y.index(max(y))])

plt.plot(x,y,linestyle='-')
plt.plot(xx,y)
plt.show()