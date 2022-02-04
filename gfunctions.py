from __future__ import division

import numpy as np

from scipy import integrate
from scipy import special
from scipy.integrate import quad
from scipy.special import erf
from scipy.special import erfc



#### Single Borehole ####
def ierf(X):
	return X*special.erf(X)-(1/np.sqrt(np.pi))*(1-np.exp(-X**2))
	
def Ils(h,d):
	return 2*ierf(h) + 2*ierf(h+2*d) - ierf(2*h+2*d) - ierf(2*d)
	
def g_FLSjc(x,y,ro,z,H,lm,Cm,t):
	# Finite Line Sourve: Javed and Claesson 2011
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]	
	g = (1.0/4.0/np.pi/lm)*(integrate.quad(lambda s,r,H,Dt,t: np.exp(-r**2 * s**2)* Ils(H*s,1*s)/(H*s**2),1/np.sqrt(4*Dt*t),np.inf,args = (r,H,Dt,t))[0])
	return g

def g_ILS(x,y,ro,z,H,lm,Cm,t):
	# (ILS) Infinite Line Source : Carslaw & Jaeger 1959 
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]	
	g = (1.0/4.0/np.pi/lm)*special.expn(1,r**2.0/4.0/Dt/t)
	return g
	
def g_ICS(x,y,ro,z,H,lm,Cm,t):
	# (ICS) Infinite Cylindrical Source : Man et al. (2010) 	
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]		
	g = (1.0/4.0/np.pi/lm)*(integrate.quad(lambda f,r,ro,Dt,t: (1/np.pi)*special.expn(1,((r**2+ro**2-2*r*ro*np.cos(f))/4.0/Dt/t)),0,np.pi,args=(r,ro,Dt,t))[0])	
	return g
	
def g_FCS(x,y,ro,z,H,lm,Cm,t):
	# (ICS) Infinite Cylindrical Source : Man et al. (2010) 	
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]		
	func = lambda f,ze,z,r,ro,Dt,t: special.erfc(np.sqrt(r**2+ro**2-2*r*ro*np.cos(f)+(z-ze)**2)/2/np.sqrt(Dt*t))/np.pi/np.sqrt(r**2+ro**2-2*r*ro*np.cos(f)+(z-ze)**2)
	g = (1.0/4.0/np.pi/lm)*(integrate.dblquad(func,0,H, lambda f:0, lambda f: np.pi, args = (z,r,ro,Dt,t))[0] - integrate.dblquad(func,-H,0, lambda f:0, lambda f: np.pi, args = (z,r,ro,Dt,t))[0])
	return g
	
def g_FullScale(x,y,ro,z,H,lm,Cm,t):
	g = g_ICS(x,y,ro,z,H,lm,Cm,t) + g_FLSjc(x,y,ro,z,H,lm,Cm,t) - g_ILS(x,y,ro,z,H,lm,Cm,t)
	return g



#### Multiple Boreholes ####
def g_Matrix(xPos,yPos,time,BheData):

	# Distances between boreholes
	nBHE = xPos.size
	gMatrix = np.zeros([nBHE,nBHE,time.size])
	for i in range(0,nBHE):
		for j in range(0,nBHE):
			r = ((xPos[i] - xPos[j])**2 + (yPos[i] - yPos[j])**2)**0.5
			if r == 0:
				gMatrix[i,j,:] = BheData['diamB']/2
			else:
				gMatrix[i,j,:] = r


	# calc gfunc for unique distances
	uniqueDistances = np.unique(gMatrix)
	gFuncs = np.zeros([uniqueDistances.size,time.size])
	for i in range(0,uniqueDistances.size):
		for j in range(0,time.size):
			gFuncs[i,j] = g_FullScale(uniqueDistances[i],0,BheData['diamB']/2,BheData['length']/2,BheData['length'],BheData['lm'],BheData['Cm'],time[j])


	# write gfuncs to matrix
	for i in range(0,nBHE):
		for j in range(0,nBHE):
			for k in range(0,uniqueDistances.size):
				if gMatrix[i,j,0] == uniqueDistances[k]:
					gMatrix[i,j,:] = gFuncs[k,:]

	return gMatrix
	

