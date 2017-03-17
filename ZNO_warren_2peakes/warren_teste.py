#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Andrezio
#
# Created:     16/03/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from math import sin,cos,pi,radians,tan,sqrt,log1p

namefile=['outsample101.xy','outstandart101.xy','outsample202.xy','outstandart202.xy']


def calc_Fourier(x,y):
    armonico=[] #numeros armonicos
    AN=[] # real

    tamanho=len(y)

    x=list(x)
    y=list(y)


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




    return AN,armonico


def calc_Fourier_img(x,y):
    armonico=[] #numeros armonicos
    AN=[] # real

    tamanho=len(y)

    x=list(x)
    y=list(y)


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
            soma = soma+yy[j]*sin(2*pi*i*Nx[j]/tamanho)


        if soma <=0:
            pass
        else:
            AN.append(soma/tamanho)
            armonico.append(i)


    return AN,armonico


def normalizar(vetor):
    maximo=max(vetor)
    newvetor=[]
    for i in vetor:
        newvetor.append(i/maximo)

    return newvetor

def bertho(x,y,x1,y1):

    AN,armonico=calc_Fourier(x,y)
    ANST,armonicoST=calc_Fourier(x1,y1)

    ANi,armonicoi=calc_Fourier_img(x,y)
    ANSTi,armonicoSTi=calc_Fourier_img(x1,y1)

    newAN=[]
    newarmonico=[]
    for i in range(len(AN)):
        try:

            cima=AN[i]*ANST[i]+ANi[i]*ANSTi[i]
            baixo=pow(ANST[i],2)+pow(ANSTi[i],2)
            newAN.append(cima/baixo)
            newarmonico.append(armonico[i])

        except:
            pass

    return newarmonico,newAN

def newxvetor(X,s):
    newvetorx=[]
    for i in range(len(X)):
        newvetorx.append(s)
    return newvetorx

x,y = np.loadtxt(namefile[0], unpack= True) #sample 101
x1,y1 = np.loadtxt(namefile[1], unpack= True) #standart 101
X,Y=bertho(x,y,x1,y1)
Y=normalizar(Y[0:15])



x2,y2 = np.loadtxt(namefile[2], unpack= True)
x3,y3 = np.loadtxt(namefile[3], unpack= True)
X1,Y1=bertho(x2,y2,x3,y3)
Y1=normalizar(Y1[0:15])



X1=newxvetor(X1,2)
X=newxvetor(X,1)



def plotar():
    for i in range(len(X[0:12])):
        try:
            x=[X[i],X1[i]]
            y=[Y[i],Y1[i]]

            coeficiente=abs((Y1[i]-Y[i])/(X1[i]-X[i]))

            print 'alpha '+ str(i) + ' '  + str(coeficiente)

            plt.plot(x,y,linestyle='-', marker='o')
        except:
            pass

plotar()
plt.axis([0.5,2.5,0,1.2])

##plt.plot(X1[0:15],Y1[0:15],linestyle='-', marker='o')
##plt.plot(X[0:15],Y[0:15],linestyle='-', marker='o')

plt.show()



