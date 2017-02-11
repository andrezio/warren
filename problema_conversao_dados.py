#-------------------------------------------------------------------------------
# Name=        module3
# Purpose=
#
# Author=      lgallego
#
# Created=     09/02/2017
# Copyright=   (c) lgallego 2017
# Licence=     <your licence>
#-------------------------------------------------------------------------------
from math import sqrt,log1p,radians,cos
#python
amplitude=   8152.00568 #+/- 94.71700 (1.16%) (init= 7627.92)
sigma=       0.02739639 #+/- 0.000414 (1.51%) (init= 0.026)
center=      36.2270503 #+/- 0.000569 (0.00%) (init= 36.22)
gamma=       0.02739639 #+/- 0.000414 (1.51%)  == 'sigma'
fwhm=        0.09866291 #+/- 0.001492 (1.51%)  == '3.6013100*sigma'
height=      1.1871e+05 #+/- 1.38e+03 (1.16%)  == '0.3989423*amplitude/max(1.e-15, sigma)'

gauss=fwhm*0.057


#origin
y0	=-233.38651	#299,50396
xc	=36.21189	#0,00524
A	=9135.68399	#459,93296
wG	=0.00567	#0,13222
wL	=0.09855	#0,01339
Fv=0.09889944134208531

xc= cos( radians(xc/2))
wL=radians(wL)

Dorigin= 1.54/(xc*wL)

fv = 0.5346*wL + sqrt(0.2166*(pow(wL,2))+pow(wG,2))

gaussian =2.3548*fwhm
