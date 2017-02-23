#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     06/01/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from Tkinter import *
from ttk import *
import sys
import tkMessageBox
from tkFileDialog   import askopenfilename
import matplotlib.pyplot as plt2
from math import sin, cos
import numpy as np
from tkFileDialog   import askopenfilename
import copy
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel
from math import sin,cos,pi,radians,tan,sqrt,log1p
from scipy import stats

p2 = Tk()
positionstandart=-1
dicstandart={}




#STANDART
def stLorentxPolarization():
    global xs,ys

    for i in range(len(ys)):
        ys[i]/=(1+pow(cos(radians( xs[i])),2))/(  cos( radians( xs[i]))*pow(sin( radians( xs[i])),2))

    stPlotar()


def diciostandart():

    global xs,ys,positionstandart,dicstandart
    positionstandart+=1
    print positionstandart
    dicstandart[positionstandart]={}
    dicstandart[positionstandart]['x']= copy.copy( xs[:])
    dicstandart[positionstandart]['y']= copy.copy( ys[:])


def returnvaluesstandart():
    global xs,ys,positionstandart,dicstandart
    print 'voltar'
    print positionstandart
    positionstandart-=1



    if  positionstandart<0:
        positionstandart=0

    print positionstandart
    xs=copy.copy(  dicstandart[positionstandart]['x'])
    ys=copy.copy(  dicstandart[positionstandart]['y'])

    stPlotarBack()


def stsavitzky_golay(y, window_size, order, deriv=0, rate=1):

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

def stnormalizar(vetor):
    maximo=max(vetor)
    newvetor=[]
    for i in vetor:
        newvetor.append(i/maximo)

    return newvetor

def stgetminmax():
    global xs,ys


    mini1=float(sbB.get())
    maxi1=float(scC.get())
    getminimo=0
    getmaximo=0
    for i in range(0, len(xs)):
        if(xs[i]<=mini1):
            try:
                getminimo=xs[i+1]
            except:
                getminimo=xs[i]
        if(xs[i]<=maxi1):
            if(xs[i]==xs[len(xs)-1]):
                getmaximo=xs[i]
            else:
                getmaximo=xs[i+1]

    mini = np.searchsorted(xs,getminimo)
    maxi = np.searchsorted(xs,getmaximo)
    return mini, maxi

def cristalmat():
    import tkMessageBox
    tkMessageBox.showinfo("CristalMat",\
    "Este e um programa gratuito\
     \ndesenvolvido e distribuido pelo grupo de pesquisa\nCristalMat -\
    IPEN\nhttp://www.cristalmat.net/")

def close_window ():
    Fechar()
    p2.destroy()

def stOpen_file():
    global xs,ys,x0s,y0s,namefile
    namefile = askopenfilename()


    try:
        xs,ys,zs = np.loadtxt(namefile, unpack= True)
        x0s=copy.copy(xs[:])
        y0s=copy.copy(ys[:])
        print 'tres colunas'

    except:
        xs,ys = np.loadtxt(namefile, unpack= True)
        x0s=copy.copy(xs[:])
        y0s=copy.copy(ys[:])
        print 'duas colunas'

    scC.delete(0,END)
    sbB.delete(0,END)
    scC.insert(1,xs[-1])
    sbB.insert(1,xs[0])

def Fechar():
    plt2.close()

def stPlotar():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global xs,ys
    mini,maxi=stgetminmax()
    try:
        diciostandart()
        plt2.cla()
        plt2.title('Amostra')
        plt2.xlabel('2Theta')
        plt2.ylabel("Intensity")
        plt2.plot(xs[mini:maxi],ys[mini:maxi],linestyle='-', marker='o')
        plt2.grid()
        plt2.show()

    except:
        print 'vazio'

def stPlotarBack():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global xs,ys
    mini,maxi=stgetminmax()
    try:

        plt2.cla()
        plt2.title('Amostra')
        plt2.xlabel('2Theta')
        plt2.ylabel("Intensity")
        plt2.plot(xs[mini:maxi],ys[mini:maxi],linestyle='-', marker='o')
        plt2.grid()
        plt2.show()

    except:
        print 'vazio'

def stResetar():
    global xs,ys
    global x0s,y0s
    xs=copy.copy(x0s[:])
    ys=copy.copy(y0s[:])

    scC.delete(0,END)
    sbB.delete(0,END)
    scC.insert(1,xs[-1])
    sbB.insert(1,xs[0])

    stPlotarBack()

def stNormalizar():

    global xs,ys
    mini,maxi=stgetminmax()

    xs=xs[mini:maxi]
    ys=ys[mini:maxi]
    ys=stnormalizar(ys)
    print 'normalizar'

    stPlotar()

def stCentralizar():
    global xs,ys
    tamanho=len(ys)
    ys=stnormalizar(ys)
    if tamanho/2>ys.index(max(ys)):
        print 'maior',tamanho/2-ys.index(max(ys))
        lados=tamanho/2-ys.index(max(ys))
    else:
        print 'menor'
        lados=-tamanho/2+ys.index(max(ys))

    indice=ys.index(max(ys))+lados

    stPlotar()


def stSuavizar():
    print "suavizar"
    global xs,ys
    mini,maxi=stgetminmax()

    p=int(spbB.get())
    w=int(swcC.get())

    sx=xs[mini:maxi]
    sy=ys[mini:maxi]

    ys=stsavitzky_golay(ys,w,p)
    stPlotar()

def stBackground():
    global xs,ys
    mini,maxi=stgetminmax()

    xs=xs[mini:maxi]
    ys=ys[mini:maxi]

    def background (n,ys,xs):

        def list(vetor):
            newvetor = []
            for i in vetor:
                newvetor.append(i)

            return newvetor

        x1=list(xs)
        ys=list(ys)
        #print 'dados:', len(x), len(y), len(x1)
        #print y[-n:]+y[:n]
        Xn=[]

        for i in x1[:n]:
            Xn.append(i)

        for i in x1[-n:]:
            Xn.append(i)


        #print len(x1[-n:]+x1[:n]), len(y[-n:]+y[:n]), len(Xn)
        mod = LinearModel()

        pars = mod.guess(ys[-n:]+ys[:n], x=Xn)
        out  = mod.fit(ys[-n:]+ys[:n], pars, x=Xn)

        m=out.values['slope']
        b=out.values['intercept']

        Z=m*xs + b
        #print 'Z: ',len(Z)
        minimo = min(Z)
        for i in range(len(Z)):
            if Z[i]<minimo:
                Z[i]=minimo

        return Z

    n=int(spbBack.get())
    ys=ys-background(n,ys,xs)

    for i in range(len(ys)):
        if i<n:
            ys[i]=0
        elif i>=len(ys)-n:
            ys[i]=0

    minimo=ys[0]
    print minimo
    for i in range(len(ys)):
        if ys[i]<=minimo:
            ys[i]=minimo

    stPlotar()

def stdoublekalpha():
    global xs,ys
    mini,maxi=stgetminmax()
    xs=xs[mini:maxi]
    ys=ys[mini:maxi]



    stPlotar()

def stDownload():
    global xs,ys
    mini,maxi=stgetminmax()
    xs=xs[mini:maxi]
    ys=ys[mini:maxi]
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f

    for i in range(len(xs)):
        print xs[i], str(';'),ys[i]
    sys.stdout = orig_stdout
    f.close()

#############################################
texto = Label(p2,text='STANDART').place(x=5,y=5)

horizontal=0
vertical=40


btnPlotar = Button(p2, text="STANDART",command = stOpen_file).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p2, text="PLOT", command = stPlotar).place(x=horizontal,y=vertical)
vertical+=30
btnResetar = Button(p2, text="RESETAR", command = stResetar).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p2, text="CLOSE", command = Fechar).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p2, text="BACK",command=returnvaluesstandart).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p2, text="DOWNLOAD",command=stDownload).place(x=horizontal,y=vertical)

texto = Label(p2,text='CORRECTION').place(x=120,y=5)

horizontal=120
vertical=40

btnNormalizar = Button(p2, text="NORMALIZE", command = stNormalizar).place(x=horizontal,y=vertical)
vertical+=30
##################################polinomios
sp=9
sw=11
horizontal_2=200

sxc = Label(p2, text = "Pol")
sxc.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=20
spbB = Entry(p2, textvariable = sp)
spbB.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )
horizontal_2+=40
sxd = Label(p2, text = "Win")
sxd.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=30
swcC = Entry(p2, textvariable = sw)
swcC.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

swcC.delete(0,END)
spbB.delete(0,END)
swcC.insert(1,int(sw))
spbB.insert(1,int(sp))
##################################polinomios


btnNormalizar = Button(p2, text="SMOOTH", command = stSuavizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p2, text="CENTRALIZE", command = stCentralizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p2, text="LORENTZPOLARIZATION",state=NORMAL,command = stLorentxPolarization).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p2, text="DOUBLETOKALPHA",state=NORMAL,command = stdoublekalpha).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p2, text="BACKGROUND",command = stBackground).place(x=horizontal,y=vertical)

spback=10
horizontal_2=210
sxc = Label(p2, text = "size")
sxc.place(bordermode = OUTSIDE, height = 30, width = 40, x =horizontal_2,y=vertical )
horizontal_2+=30
spbBack = Entry(p2, textvariable = spback)
spbBack.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

spbBack.delete(0,END)
spbBack.insert(1,int(spback))


ak=260
texto = Label(p2,text='POSITION').place(x=50,y=ak-30)
sa=int
sb=int
sxc = Label(p2, text = "Min")
sxc.place(bordermode = OUTSIDE, height = 30, width = 30, x =0,y=ak )

sbB = Entry(p2, textvariable = sa)
sbB.place(bordermode = OUTSIDE, height = 30, width = 40, x = 30, y =ak )

sxd = Label(p2, text = "Max")
sxd.place(bordermode = OUTSIDE, height = 30, width = 30, x =70,y=ak )

scC = Entry(p2, textvariable = sb)
scC.place(bordermode = OUTSIDE, height = 30, width = 50, x = 100, y =ak )
#########################################
###################################

#janela----

p2.title("Cristal Mat - IPEN")
p2.geometry("650x330+10+10")
p2.mainloop()