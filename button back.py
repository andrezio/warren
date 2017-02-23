#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lgallego
#
# Created:     23/02/2017
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

dicio={}
inicio=0

namefile = "ZnO_DEF_101_F_b_ka2.asc"


def savitzky_golay(y, window_size=21, order=9, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


x,y = np.loadtxt(namefile, unpack= True)


def diciof(positin):
    global x,y
    dicio[positin]={}
    dicio[positin]['x']=x
    dicio[positin]['y']=y

diciof(inicio)
inicio+=1


def soma():
    global x,y,inicio,dicio
    diciof(inicio)
    inicio+=1
    plt.cla()
    y=savitzky_golay(y)
    plt.plot(x,y)
    plt.show()

def returnvalues():
    global dicio,inicio,x,y
    if inicio==0 or inicio<0:
        inicio=0
    x=dicio[inicio]['x']
    y=dicio[inicio]['y']

def varsoltar():
    global inicio,x,y
    inicio-=1
    returnvalues()
    plt.cla()
    plt.plot(x,y)
    plt.show()

root = Tk()


btnCentralizar = Button(root, text="BACKGROUND",command = soma).place(x=10,y=10)
btnCentralizar = Button(root, text="VOLTAR",command = varsoltar).place(x=10,y=50)


root.title("Cristal Mat - IPEN")
root.geometry("650x330+10+10")
root.mainloop()