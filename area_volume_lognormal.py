#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      Andrezio
#
# Created:     19/02/2017
# Copyright:   (c) Andrezio 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#calculo log-normal
from math import exp,log,sqrt,pi
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

e=2.71828182845904523536028747135266249775724709

Larea=49.5082
Lvolume=94.6759

Dzero=0

Dzero = (8*Lvolume)/(9*Larea)

Dzero = log(Dzero,e)

Dzero = sqrt(Dzero)

Dzero = exp(Dzero)

Sigma = Dzero

Dzero = log(Dzero,e)

Dzero = pow(Dzero, 2)

Dzero = -1*(5/2)*Dzero

Dzero = exp(Dzero)

Dzero = Dzero*(3/2)*Larea

##def seq(start, stop, step):
##    n = int(round((stop - start)/float(step)))
##    if n > 1:
##        return([start + step*i for i in range(n+1)])
##    else:
##        return([])
##
##x=seq(0,100,0.1)
##
##pdf=[]
##for i in x:
##    valor = 0
##    try:
##        valor = log(i/Dzero,e)
##    except:
##        valor = 0
##    valor = valor/log(Sigma,e)
##    valor = valor*(-1)*(0.5)
##    valor=exp(valor)
##    try:
##       valor2 = 1/(sqrt(2*pi*pow(i,2)*pow(log(Sigma,e),2)))
##    except:
##       valor2 =0
##
##    valor= valor*valor2
##
##    pdf.append(   valor   )
##
##
##plt.plot(x,pdf,linestyle='-', marker='o')
##plt.show()