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
import Pmw

import sys
import tkMessageBox
from tkFileDialog   import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from math import sin, cos
import numpy as np
from tkFileDialog   import askopenfilename
import copy
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel
from math import sin,cos,pi,radians,tan,sqrt,log1p
from scipy import stats

root = Tk()
root.title('Notebook')
Pmw.initialise()

positionsampl=-1
positionstandart=-1
dicsample={}
dicstandart={}

##############
#SAMPLE
def diciosample():

    global x,y,positionsampl,dicsample
    positionsampl+=1
    dicsample[positionsampl]={}
    dicsample[positionsampl]['x']= copy.copy( x)
    dicsample[positionsampl]['y']= copy.copy( y)


def returnvaluessample():
    print 'voltar'
    positionsampl-=1

    global dicsample,positionsampl,x,y

    if  positionsampl<0:
        positionsampl=0

    x=copy.copy(  dicsample[positionsampl]['x'])
    y=copy.copy(  dicsample[positionsampl]['y'])
    PlotarBack()
#STANDART
##def diciostandart():
##
##    global x,y,positionstandart,dicstandart
##    positionstandart+=1
##    dicsample[positionstandart]={}
##    dicsample[positionstandart]['x']= copy.copy( xs)
##    dicsample[positionstandart]['y']= copy.copy( ys)
##
##
##def returnvaluesstandart():
##    print 'voltar'
##    positionstandart-=1
##
##    global x,y,positionstandart,dicstandart
##
##    if  positionstandart<0:
##        positionstandart=0
##
##    x=copy.copy(  dicsample[positionstandart]['x'])
##    y=copy.copy(  dicsample[positionstandart]['y'])
##    PlotarBack()

def Download():
    global x,y
    mini,maxi=tgetminmax()
    x=x[mini:maxi]
    y=y[mini:maxi]
    orig_stdout = sys.stdout
    f = open('outsample.txt', 'w')
    sys.stdout = f

    for i in range(len(x)):
        print x[i], str(';'),y[i]
    sys.stdout = orig_stdout
    f.close()

def removekalpha():
    pass

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

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

def normalizar(vetor):
    maximo=max(vetor)
    newvetor=[]
    for i in vetor:
        newvetor.append(i/maximo)

    return newvetor

def getminmax():
    global x,y


    mini1=float(bB.get())
    maxi1=float(cC.get())
    getminimo=0
    getmaximo=0
    for i in range(0, len(x)):
        if(x[i]<=mini1):
            try:
                getminimo=x[i+1]
            except:
                getminimo=x[i]
        if(x[i]<=maxi1):
            if(x[i]==x[len(x)-1]):
                getmaximo=x[i]
            else:
                getmaximo=x[i+1]

    mini = np.searchsorted(x,getminimo)
    maxi = np.searchsorted(x,getmaximo)
    return mini, maxi




def cristalmat():
    import tkMessageBox
    tkMessageBox.showinfo("CristalMat",\
    "Este e um programa gratuito\
     \ndesenvolvido e distribuido pelo grupo de pesquisa\nCristalMat -\
    IPEN\nhttp://www.cristalmat.net/")

def close_window ():
    Fechar()
    root.destroy()

def Open_file():
    global x,y,x0,y0,namefile
    namefile = askopenfilename()


    try:
        x,y,z = np.loadtxt(namefile, unpack= True)
        x0=copy.copy(x[:])
        y0=copy.copy(y[:])
        print 'tres colunas'

    except:
        x,y = np.loadtxt(namefile, unpack= True)
        x0=copy.copy(x[:])
        y0=copy.copy(y[:])
        print 'tres colunas'

    cC.delete(0,END)
    bB.delete(0,END)
    cC.insert(1,x[-1])
    bB.insert(1,x[0])

    diciosample()

def Open_file_standart():
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
        print 'tres colunas'

    cC.delete(0,END)
    bB.delete(0,END)
    cC.insert(1,xs[-1])
    bB.insert(1,xs[0])

    diciostandart()

def Resetar_standart():
    global xs,ys
    global x0s,y0s
    xs=copy.copy(x0s[:])
    ys=copy.copy(y0s[:])

    cC.delete(0,END)
    bB.delete(0,END)
    cC.insert(1,xs[-1])
    bB.insert(1,xs[0])

    PlotarStandart()

def Fechar():
    plt.close()

def Plotar():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global x,y
    mini,maxi=getminmax()
    try:
        diciosample()
        plt.cla()
        plt.title('Amostra')
        plt.xlabel('2Theta')
        plt.ylabel("Intensity")
        plt.plot(x[mini:maxi],y[mini:maxi],linestyle='-', marker='o')
        plt.grid()
        plt.show()

    except:
        print 'vazio'

def PlotarBack():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global x,y
    mini,maxi=getminmax()
    try:

        plt.cla()
        plt.title('Amostra')
        plt.xlabel('2Theta')
        plt.ylabel("Intensity")
        plt.plot(x[mini:maxi],y[mini:maxi],linestyle='-', marker='o')
        plt.grid()
        plt.show()

    except:
        print 'vazio'

def Plotarstandart():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global xs,ys
    mini,maxi=getminmax()
    try:
        diciosample()
        plt.cla()
        plt.title('Amostra')
        plt.xlabel('2Theta')
        plt.ylabel("Intensity")
        plt.plot(xs[mini:maxi],ys[mini:maxi],linestyle='-', marker='o')
        plt.grid()
        plt.show()

    except:
        print 'vazio'

def Resetar():
    global x,y
    global x0,y0
    x=copy.copy(x0[:])
    y=copy.copy(y0[:])

    cC.delete(0,END)
    bB.delete(0,END)
    cC.insert(1,x[-1])
    bB.insert(1,x[0])

    Plotar()

def Normalizar():

    global x,y
    mini,maxi=getminmax()

    x=x[mini:maxi]
    y=y[mini:maxi]
    y=normalizar(y)
    print 'normalizar'

    Plotar()

def Centralizar():
    global x,y
    tamanho=len(y)
    y=normalizar(y)
    if tamanho/2>y.index(max(y)):
        print 'maior',tamanho/2-y.index(max(y))
        lados=tamanho/2-y.index(max(y))
    else:
        print 'menor'
        lados=-tamanho/2+y.index(max(y))

    indice=y.index(max(y))+lados

    Plotar()

def radiation(key):
    dic={"W - 0.0209(nm)":0.0209,
"Mo - 0.0709(nm)":0.0709,	"Cu - 0.154(nm)":0.154,	"Ag - 0.0559(nm)":0.0559,\
	"Ga - 0.134(nm)":0.134,	"In - 0.0512(nm)":0.0512
    }

    return dic[key]


def SingleLine():
    plt.close()
    global x,y
    mini,maxi=getminmax()

    x=x[mini:maxi]
    y=y[mini:maxi]


    if str(comboBox.get())=='VoigtModel':
        mod = VoigtModel()
    elif str(comboBox.get())=='PseudoVoigtModel':
        mod = PseudoVoigtModel()

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)



    print(out.fit_report())

    plt.figure(1)

    plt.subplot(221)
    plt.plot(x, y,label='original data',linestyle='-', marker='o')
    plt.title('Amostra')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    ##plt.plot(x, out.init_fit, 'k--',label='initial ')

    plt.subplot(222)
    plt.plot(x, out.best_fit, 'r-',label='best fit',linestyle='-', marker='o')
    plt.title('Amostra')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    plt.subplot(212)

    plt.plot(x,y,linestyle='-', marker='o')
    plt.title(str(str(comboBox.get())))




    lambida=radiation(comboBoxrad.get())

    #D=(lambida)/(  radians( out.best_values['sigma']*0.5*sqrt(pi/log1p(2))) *2*cos( radians( out.best_values['center']/2)))

    center=out.best_values['center']/2
    center=radians(center)
    center=cos(center)

    sigmaL=out.best_values['sigma']*3.6013100
    sigmaL=radians(sigmaL)


    D = lambida/(sigmaL*center)


    E=2.35482*( radians( out.best_values['sigma']*pi/2 ))/(4*tan(radians( out.best_values['center']/2)))

    if E<0:
        E*=-1
    if D<0:
        D*=-1



    t = plt.text(0.5, 0.5, '$L_V(nm)$: '+ str(D) + '\n$<e>$: '+ str(E), transform=plt.subplot(212).transAxes, fontsize=10)
    t.set_bbox(dict(color='red', alpha=0.5, edgecolor='red'))


    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.plot(x, out.best_fit, 'r-',label='center: '+str(out.best_values['center'])+'\nSigma: '+ str(out.best_values['sigma']),linestyle='-', marker='o')
    plt.legend()
    plt.grid()
    plt.show()

def LorentxPolarization():
    global x,y

    for i in range(len(y)):
        y[i]/=(1+pow(cos(radians( x[i])),2))/(  cos( radians( x[i]))*pow(sin( radians( x[i])),2))

    Plotar()

def Suavizar():
    print "suavizar"
    global x,y
    mini,maxi=getminmax()

    p=int(pbB.get())
    w=int(wcC.get())

    x=x[mini:maxi]
    y=y[mini:maxi]

    y=savitzky_golay(y,w,p)
    Plotar()

def Background():
    global x,y
    mini,maxi=getminmax()

    x=x[mini:maxi]
    y=y[mini:maxi]

    def background (n,y,x):

        def list(vetor):
            newvetor = []
            for i in vetor:
                newvetor.append(i)

            return newvetor

        x1=list(x)
        y=list(y)
        #print 'dados:', len(x), len(y), len(x1)
        #print y[-n:]+y[:n]
        Xn=[]

        for i in x1[:n]:
            Xn.append(i)

        for i in x1[-n:]:
            Xn.append(i)


        #print len(x1[-n:]+x1[:n]), len(y[-n:]+y[:n]), len(Xn)
        mod = LinearModel()

        pars = mod.guess(y[-n:]+y[:n], x=Xn)
        out  = mod.fit(y[-n:]+y[:n], pars, x=Xn)

        m=out.values['slope']
        b=out.values['intercept']

        Z=m*x + b
        #print 'Z: ',len(Z)
        minimo = min(Z)
        for i in range(len(Z)):
            if Z[i]<minimo:
                Z[i]=minimo

        return Z

    n=int(pbBack.get())
    y=y-background(n,y,x)

    for i in range(len(y)):
        if i<n:
            y[i]=0
        elif i>=len(y)-n:
            y[i]=0

    minimo=y[0]
    print minimo
    for i in range(len(y)):
        if y[i]<=minimo:
            y[i]=minimo

    Plotar()


def Fourier():
    plt.close()
    global x,y

    x1=copy.copy(x)
    y1=copy.copy(y)

    mini,maxi=getminmax()

    x=x[mini:maxi]
    y=y[mini:maxi]

    armonico=[] #numeros armonicos
    AN=[] # real



    tamanho=len(y)

    def list(vetor):
            newvetor = []
            for i in vetor:
                newvetor.append(i)

            return newvetor

    x=list(x)
    y=list(y)

    a=[]
    for i in range(0,21):
        a.append(0)

    Nx=[]
    for i in range(-1*y.index(max(y)),y.index(max(y))):
        Nx.append(i)

    primeiro=0
    maior=0
    menor=0
    for i in x:

        if primeiro ==0:
            if not i==0:
                menor=i
                primeiro=1

        if primeiro ==1:
            if i==0:
                primeiro=2
            if primeiro==1:
                maior=i


    yy=[]
    for i in range(len(Nx)):
        try:
            yy.append(y[i])
        except:
            pass


    for i in range(len(yy)):
        #armonico.append(i)
        soma=0

        for j in range(len(yy)-1):
            soma = soma+yy[j]*cos(2*pi*i*Nx[j]/tamanho)


        if soma <=0:
            pass
        else:
            AN.append(soma/tamanho)
            armonico.append(i)


    AN=normalizar(AN)

    lambida=radiation(comboBoxrad.get())

    print menor, maior, sin(menor),sin(maior)

##    menor=radians(menor)
##    maior=radians(maior)
    menor=radians(menor/2)
    maior=radians(maior/2)

    for i in range(len(armonico)):
        armonico[i]=(i*lambida)/((sin(maior)-sin(menor))*2)

    for i in range(len(armonico)):
        if armonico[i]<0:
            armonico[i]*=-1


    plt.figure(1)

    plt.subplot(221)
    plt.grid()
    plt.xlabel('position (2theta)')
    plt.ylabel("Intensity")
    plt.title("Amostra de ZnO")
    plt.plot(x,y,linestyle='-', marker='o')

    plt.subplot(222)
    plt.grid()
    plt.plot(armonico[0:30],AN[0:30], c='k',linestyle='-', marker='o')
    plt.xlabel('L(nm)')
    plt.ylabel("A(L)")
    plt.title("Amostra de ZnO - Fourier ")

    inicio=int(boxFmin.get())
    fim=int(boxFmax.get())

    y=AN[inicio:fim]
    x=armonico[inicio:fim]

    mod = LinearModel()

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    XS=out.values['intercept']/out.values['slope']*-1


    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)


    plt.subplot(212)
    plt.grid()
    plt.plot(armonico[0:30],AN[0:30],linestyle='--', marker='o')
    plt.plot(x, out.best_fit, 'r-', label='$L_A(nm)$: '+str("%.4f"%XS))
    plt.xlabel('L(nm)')
    plt.ylabel("A(L)")
    plt.title("Amostra de ZnO - Fourier ")
    plt.legend()

    # Create a list of values in the best fit line
    abline_values = [slope * i + intercept for i in armonico]

    lx=[]
    ly=[]
    for i in range(len(abline_values)):
        if abline_values[i]>=0:
            lx.append(armonico[i])
            ly.append(abline_values[i])

    plt.plot(lx,ly, 'red')
    x=x1
    y=y1
    plt.show()
##############

nb = Pmw.NoteBook(root)
p1 = nb.add('SAMPLE')
p2 = nb.add('STANDART')
p3 = nb.add('ANALYSIS')

nb.pack(padx=5, pady=5, fill=BOTH, expand=1)

#P1
#Button(p1, text='This is text on page 1', fg='blue').pack(pady=40)
texto = Label(p1,text='SHOW').place(x=10,y=5)

horizontal=0
vertical=40

btnPlotar = Button(p1, text="SAMPLE",command=Open_file).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p1, text="PLOT", command = Plotar).place(x=horizontal,y=vertical)
vertical+=30
btnResetar = Button(p1, text="RESETAR", command = Resetar).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p1, text="CLOSE", command = Fechar).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p1, text="BACK", command = returnvaluessample).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(p1, text="DOWNLOAD", command = Download).place(x=horizontal,y=vertical)


texto = Label(p1,text='CORRECTION').place(x=120,y=5)

horizontal=120
vertical=40

btnNormalizar = Button(p1, text="NORMALIZE", command = Normalizar).place(x=horizontal,y=vertical)
vertical+=30
##################################polinomios
p=9
w=11
horizontal_2=200

xc = Label(p1, text = "Pol")
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=20
pbB = Entry(p1, textvariable = p)
pbB.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )
horizontal_2+=40
xd = Label(p1, text = "Win")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=30
wcC = Entry(p1, textvariable = w)
wcC.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

wcC.delete(0,END)
pbB.delete(0,END)
wcC.insert(1,int(w))
pbB.insert(1,int(p))

btnNormalizar = Button(p1, text="SMOOTH", command = Suavizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p1, text="CENTRALIZE", command = Centralizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p1, text="LORENTZPOLARIZATION",state=NORMAL,command = LorentxPolarization).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p1, text="DOUBLETOKALPHA",command = removekalpha).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(p1, text="BACKGROUND",command = Background).place(x=horizontal,y=vertical)

pback=10
horizontal_2=210
xc = Label(p1, text = "size")
xc.place(bordermode = OUTSIDE, height = 30, width = 40, x =horizontal_2,y=vertical )
horizontal_2+=30
pbBack = Entry(p1, textvariable = pback)
pbBack.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

pbBack.delete(0,END)
pbBack.insert(1,int(pback))

ak=260
texto = Label(p1,text='POSITION').place(x=50,y=ak-30)
a=int
b=int
xc = Label(p1, text = "Min")
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =0,y=ak )

bB = Entry(p1, textvariable = a)
bB.place(bordermode = OUTSIDE, height = 30, width = 40, x = 30, y =ak )

xd = Label(p1, text = "Max")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =70,y=ak )

cC = Entry(p1, textvariable = b)
cC.place(bordermode = OUTSIDE, height = 30, width = 50, x = 100, y =ak )

######################
#P2


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
##    global xs,ys
##    mini,maxi=stgetminmax()
##    xs=xs[mini:maxi]
##    ys=ys[mini:maxi]
##
##
##
##    stPlotar()
    pass

#############################################

#===========================single line double
def SingleLineDouble():
    plt.close()
    global x,y,xs,ys
    mini,maxi=getminmax()
    mini,maxi=stgetminmax()

    if str(comboBox.get())=='VoigtModel':
        mod = VoigtModel()
    elif str(comboBox.get())=='PseudoVoigtModel':
        mod = PseudoVoigtModel()

    x=x[mini:maxi]
    y=y[mini:maxi]

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)

    xs=xs[mini:maxi]
    ys=ys[mini:maxi]

    pars1 = mod.guess(ys, x=xs)
    out1  = mod.fit(ys, pars1, x=xs)






    #print(out.fit_report())

    plt.figure(1)

    plt.subplot(221)
    plt.plot(x, y,label='sample data',linestyle='-', marker='o')
    plt.plot(xs, ys,label='standart data',linestyle='-', marker='o')
    plt.title('Amostra')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    ##plt.plot(x, out.init_fit, 'k--',label='initial ')

    plt.subplot(222)
    plt.plot(x, out.best_fit, 'r-',label='best sample fit',linestyle='-', marker='o')
    plt.plot(xs, out1.best_fit, 'r-',label='best standart fit',linestyle='-', marker='o')
    plt.title('FIT')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    plt.subplot(212)

    plt.plot(x,y,linestyle='-', marker='o')
    plt.plot(xs,ys,linestyle='-', marker='o')
    plt.title(str(str(comboBox.get())))




    lambida=radiation(comboBoxrad.get())

    #D=(lambida)/(  radians( out.best_values['sigma']*0.5*sqrt(pi/log1p(2))) *2*cos( radians( out.best_values['center']/2)))

    #center=out.best_values['center']/2
    #print out.best_values['center'] , out1.best_values['center']
    center=(out.best_values['center'] + out1.best_values['center'] )/2
    #print center
    center = center /2

    center=radians(center)

    center=cos(center)

    sigmaL=(out.best_values['sigma']-out1.best_values['sigma'] )*3.6013100

    sigmaL=radians(sigmaL)

    #print center, sigmaL

    D = lambida/(sigmaL*center)


    E=2.35482*( radians( (out.best_values['sigma']-out1.best_values['sigma'] )*pi/2 ))/(4*tan(radians( (out.best_values['sigma']-out1.best_values['sigma'] )/2)))

    if E<0:
        E*=-1
    if D<0:
        D*=-1



    t = plt.text(0.5, 0.5, '$L_V(nm)$: '+ str(D) + '\n$<e>$: '+ str(E), transform=plt.subplot(212).transAxes, fontsize=10)
    t.set_bbox(dict(color='red', alpha=0.5, edgecolor='red'))


    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.plot(x, out.best_fit, 'r-',label='center: '+str(out.best_values['center'])+'\nSigma: '+ str(out.best_values['sigma']),linestyle='-', marker='o')
    plt.legend()
    plt.grid()
    plt.show()

#===================================================

def LorentxPolarization():
    global x,y

    for i in range(len(y)):
        y[i]/=(1+pow(cos(radians( x[i])),2))/(  cos( radians( x[i]))*pow(sin( radians( x[i])),2))

    Plotar()

def Suavizar():
    print "suavizar"
    global x,y
    mini,maxi=getminmax()

    p=int(pbB.get())
    w=int(wcC.get())

    x=x[mini:maxi]
    y=y[mini:maxi]

    y=savitzky_golay(y,w,p)
    Plotar()

def Background():
    global x,y
    mini,maxi=getminmax()

    x=x[mini:maxi]
    y=y[mini:maxi]

    def background (n,y,x):

        def list(vetor):
            newvetor = []
            for i in vetor:
                newvetor.append(i)

            return newvetor

        x1=list(x)
        y=list(y)
        #print 'dados:', len(x), len(y), len(x1)
        #print y[-n:]+y[:n]
        Xn=[]

        for i in x1[:n]:
            Xn.append(i)

        for i in x1[-n:]:
            Xn.append(i)


        #print len(x1[-n:]+x1[:n]), len(y[-n:]+y[:n]), len(Xn)
        mod = LinearModel()

        pars = mod.guess(y[-n:]+y[:n], x=Xn)
        out  = mod.fit(y[-n:]+y[:n], pars, x=Xn)

        m=out.values['slope']
        b=out.values['intercept']

        Z=m*x + b
        #print 'Z: ',len(Z)
        minimo = min(Z)
        for i in range(len(Z)):
            if Z[i]<minimo:
                Z[i]=minimo

        return Z

    n=int(pbBack.get())
    y=y-background(n,y,x)

    for i in range(len(y)):
        if i<n:
            y[i]=0
        elif i>=len(y)-n:
            y[i]=0

    minimo=y[0]
    print minimo
    for i in range(len(y)):
        if y[i]<=minimo:
            y[i]=minimo

    Plotar()


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


#P3
########################
horizontal=5
texto = Label(p3,text='ANALYSIS SAMPLE').place(x=horizontal,y=5)


vertical=40


btnSingleLine = Button(p3,  text="SINGLE LINE", command = SingleLine).place(x=horizontal,y=vertical)
#,state = DISABLED
vertical+=30
btnFourier = Button(p3,  text="FOURIER", command=Fourier).place(x=horizontal,y=vertical)

btnFLognormal = Button(p3,  text="LOG-NORMAL").place(x=horizontal,y=vertical+30)
#########################################
ak=vertical
##Fmin=int
##Fmax=int
Fmin=1
Fmax=5

xc = Label(p3, text = "Min")
beta=horizontal+80
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmin = Entry(p3, textvariable = Fmin)
boxFmin.place(bordermode = OUTSIDE, height = 30, width = 40, x = beta, y =ak )
beta+=40

xd = Label(p3, text = "Max")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmax = Entry(p3, textvariable = Fmax)
boxFmax.place(bordermode = OUTSIDE, height = 30, width = 50, x = beta, y =ak )

boxFmin.delete(0,END)
boxFmax.delete(0,END)
boxFmin.insert(1,int(Fmin))
boxFmax.insert(1,int(Fmax))

#########################################
########################
horizontal=350
texto = Label(p3,text='ANALYSIS SAMPLE AND STANDART').place(x=horizontal,y=5)


vertical=40


btnSingleLine = Button(p3,  text="SINGLE LINE", command = SingleLineDouble).place(x=horizontal,y=vertical)
#,state = DISABLED
vertical+=30
btnFourier = Button(p3,  text="FOURIER", command=Fourier).place(x=horizontal,y=vertical)

btnFLognormal = Button(p3,  text="LOG-NORMAL").place(x=horizontal,y=vertical+30)
#########################################
ak=vertical
##Fmin=int
##Fmax=int
Fmin=1
Fmax=5

xc = Label(p3, text = "Min")
beta=horizontal+80
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmin = Entry(p3, textvariable = Fmin)
boxFmin.place(bordermode = OUTSIDE, height = 30, width = 40, x = beta, y =ak )
beta+=40

xd = Label(p3, text = "Max")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmax = Entry(p3, textvariable = Fmax)
boxFmax.place(bordermode = OUTSIDE, height = 30, width = 50, x = beta, y =ak )

boxFmin.delete(0,END)
boxFmax.delete(0,END)
boxFmin.insert(1,int(Fmin))
boxFmax.insert(1,int(Fmax))


#,state = DISABLED
##########################
horizontal=350
vertical=40
horizontal+=80
vertical+=2
def defocus(event):
    event.widget.master.focus_set()

##def printtext():
##    print comboBox.get()

comboBox1 = Combobox(p3, state="readonly", values=("VoigtModel", "PseudoVoigtModel"))
comboBox1.grid()
comboBox1.set("VoigtModel")
comboBox1.place(x=horizontal,y=vertical)
comboBox1.bind("<FocusIn>", defocus)



horizontal=90
vertical=42
comboBox = Combobox(p3, state="readonly", values=("VoigtModel", "PseudoVoigtModel"))
comboBox.grid()
comboBox.set("VoigtModel")
comboBox.place(x=horizontal,y=vertical)
comboBox.bind("<FocusIn>", defocus)

########################

ak=240
Labelradiation = Label(p3,text = 'RADIATION').place(x=0,y=ak+40)

comboBoxrad = Combobox(p3, state="readonly", values=("W - 0.0209(nm)",    \
"Mo - 0.0709(nm)",	"Cu - 0.154(nm)",	"Ag - 0.0559(nm)",	"Ga - 0.134(nm)",	"In - 0.0512(nm)"))


comboBoxrad.grid()
comboBoxrad.set("Cu - 0.154(nm)")
comboBoxrad.place(x=70,y=ak+40)
comboBoxrad.bind("<FocusIn>", defocus)

####################################

#menu
menubar = Menu(root)
filemenu= Menu(menubar)
filemenu.add_command(label="Open File",command=Open_file)
filemenu.add_command(label="Close",command=close_window)
filemenu.add_separator()

menubar.add_cascade(label="File",menu=filemenu)
helpmenu = Menu(menubar)
helpmenu.add_command(label="Help Index")
helpmenu.add_command(label="About", command=cristalmat)
menubar.add_cascade(label="Help",menu=helpmenu)
root.config(menu=menubar)

root.title("Cristal Mat - IPEN")
root.geometry("650x380+10+10")
root.mainloop()