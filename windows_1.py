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
import matplotlib.pyplot as plt
from math import sin, cos
import numpy as np
from tkFileDialog   import askopenfilename
import copy
from lmfit.models import VoigtModel,PseudoVoigtModel, LinearModel
from math import sin,cos,pi,radians,tan,sqrt,log1p
from scipy import stats

root = Tk()

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

def Fechar():
    plt.close()

def Plotar():

    #btnSingleLine.state==ENABLE
    #btnSingleLine.config(state=ENABLE)

    global x,y,namefile
    mini,maxi=getminmax()
    try:
        plt.cla()
        plt.title('Amostra')
        plt.xlabel('2Theta')
        plt.ylabel("Intensity")
        plt.plot(x[mini:maxi],y[mini:maxi])
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
    plt.plot(x, y,label='original data')
    plt.title('Amostra')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    ##plt.plot(x, out.init_fit, 'k--',label='initial ')

    plt.subplot(222)
    plt.plot(x, out.best_fit, 'r-',label='best fit')
    plt.title('Amostra')
    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()

    plt.subplot(212)

    plt.plot(x,y)
    plt.title(str(str(comboBox.get())))




    lambida=radiation(comboBoxrad.get())

    D=(lambida)/(  radians( out.best_values['sigma']*0.5*sqrt(pi/log1p(2))) *2*cos( radians( out.best_values['center']/2)))
    E=2.35482*( radians( out.best_values['sigma']*pi/2 ))/(4*tan(radians( out.best_values['center']/2)))

    if E<0:
        E*=-1
    if D<0:
        D*=-1



    t = plt.text(0.5, 0.5, '$L_V(nm)$: '+ str(D) + '\n$<e>$: '+ str(E), transform=plt.subplot(212).transAxes, fontsize=10)
    t.set_bbox(dict(color='red', alpha=0.5, edgecolor='red'))


    plt.xlabel('2Theta')
    plt.ylabel("Intensity")
    plt.plot(x, out.best_fit, 'r-',label='center: '+str(out.best_values['center'])+'\nSigma: '+ str(out.best_values['sigma']))
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
    plt.plot(x,y)

    plt.subplot(222)
    plt.grid()
    plt.plot(armonico[0:30],AN[0:30], c='k')
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
#############################################
texto = Label(text='PLOT').place(x=10,y=5)

horizontal=0
vertical=40

btnPlotar = Button(root, text="SAMPLE",command=Open_file).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(root, text="INSTRUMENTAL").place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(root, text="PLOT", command = Plotar).place(x=horizontal,y=vertical)
vertical+=30
btnResetar = Button(root, text="RESETAR", command = Resetar).place(x=horizontal,y=vertical)
vertical+=30
btnPlotar = Button(root, text="CLOSE", command = Fechar).place(x=horizontal,y=vertical)


texto = Label(text='CORRECTION').place(x=120,y=5)

horizontal=120
vertical=40

btnNormalizar = Button(root, text="NORMALIZE", command = Normalizar).place(x=horizontal,y=vertical)
vertical+=30
##################################polinomios
p=9
w=11
horizontal_2=200

xc = Label(root, text = "Pol")
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=20
pbB = Entry(root, textvariable = p)
pbB.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )
horizontal_2+=40
xd = Label(root, text = "Win")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =horizontal_2,y=vertical )
horizontal_2+=30
wcC = Entry(root, textvariable = w)
wcC.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

wcC.delete(0,END)
pbB.delete(0,END)
wcC.insert(1,int(w))
pbB.insert(1,int(p))
##################################polinomios


btnNormalizar = Button(root, text="SMOOTH", command = Suavizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(root, text="CENTRALIZE", command = Centralizar).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(root, text="LORENTZPOLARIZATION",state=NORMAL,command = LorentxPolarization).place(x=horizontal,y=vertical)
vertical+=30
btnCentralizar = Button(root, text="BACKGROUND",command = Background).place(x=horizontal,y=vertical)

pback=20
horizontal_2=210
xc = Label(root, text = "size")
xc.place(bordermode = OUTSIDE, height = 30, width = 40, x =horizontal_2,y=vertical )
horizontal_2+=30
pbBack = Entry(root, textvariable = pback)
pbBack.place(bordermode = OUTSIDE, height = 30, width = 40, x = horizontal_2, y =vertical )

pbBack.delete(0,END)
pbBack.insert(1,int(pback))

texto = Label(text='ANALYSIS').place(x=360,y=5)

horizontal=360
vertical=40


btnSingleLine = Button(root,  text="SINGLE LINE", command = SingleLine).place(x=horizontal,y=vertical)
#,state = DISABLED
vertical+=30
btnFourier = Button(root,  text="FOURIER", command=Fourier).place(x=horizontal,y=vertical)
#########################################
ak=vertical
##Fmin=int
##Fmax=int
Fmin=1
Fmax=5

xc = Label(root, text = "Min")
beta=horizontal+80
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmin = Entry(root, textvariable = Fmin)
boxFmin.place(bordermode = OUTSIDE, height = 30, width = 40, x = beta, y =ak )
beta+=40

xd = Label(root, text = "Max")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =beta,y=ak )
beta+=30

boxFmax = Entry(root, textvariable = Fmax)
boxFmax.place(bordermode = OUTSIDE, height = 30, width = 50, x = beta, y =ak )

boxFmin.delete(0,END)
boxFmax.delete(0,END)
boxFmin.insert(1,int(Fmin))
boxFmax.insert(1,int(Fmax))

#########################################


#,state = DISABLED
##########################
horizontal=360
vertical=40
horizontal+=80
vertical+=2
def defocus(event):
    event.widget.master.focus_set()

def printtext():
    print comboBox.get()

comboBox = Combobox(root, state="readonly", values=("VoigtModel", "PseudoVoigtModel"))
comboBox.grid()
comboBox.set("VoigtModel")
comboBox.place(x=horizontal,y=vertical)
comboBox.bind("<FocusIn>", defocus)
########################
ak=240
texto = Label(text='POSITION').place(x=50,y=ak-30)
a=int
b=int
xc = Label(root, text = "Min")
xc.place(bordermode = OUTSIDE, height = 30, width = 30, x =0,y=ak )

bB = Entry(root, textvariable = a)
bB.place(bordermode = OUTSIDE, height = 30, width = 40, x = 30, y =ak )

xd = Label(root, text = "Max")
xd.place(bordermode = OUTSIDE, height = 30, width = 30, x =70,y=ak )

cC = Entry(root, textvariable = b)
cC.place(bordermode = OUTSIDE, height = 30, width = 50, x = 100, y =ak )

Labelradiation = Label(text = 'RADIATION').place(x=0,y=ak+40)

comboBoxrad = Combobox(root, state="readonly", values=("W - 0.0209(nm)",    \
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





#janela----

root.title("Cristal Mat - IPEN")
root.geometry("650x330+10+10")
root.mainloop()