from __future__ import division
#import time
import numpy as np

from scipy.fft import fft, ifft
import os


from bhe_models import*
from gfunctions import*

	
def func_lin(x,k,n):
	return x*k+n
			
def func_lin0(x,k):
	return x*k

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
		
# Fourier #
def Fourier(a):
	return np.fft.fft(np.concatenate((a,np.zeros(a.size-1))))
	
def invFourier(a):
	return np.real(np.fft.ifft(a))[0:int((a.size+1)/2)]
	

def FourierSci(a):
	return fft(np.concatenate((a,np.zeros(a.size-1))))
	
def invFourierSci(a):
	return np.real(ifft(a))[0:int((a.size+1)/2)]	
	

def opt_nper(nt,psi_sa,psi_FFT,c_sa):
	""" Calculate optimal number of periods for hybrid model
	inputs:	nt: total number of timesteps
			psi_sa: coefficient for semi analytical model
			c_sa:	coefficient for semi analytical model
			psi_FFT:coefficient for FFT model
	"""
	return np.sqrt(psi_sa*nt+c_sa-2*psi_FFT)/np.sqrt(psi_FFT)


def mean_params(nsp,nb,psi_sa_l,psi_sa_h,c_sa_l,c_sa_h):
	""" Calculate mean coefficients if number of steps per period is above nb
	inputs:	nsp:		number of timesteps per period
			nb:			breakpoint
			psi_sa_l:	coefficient for semianalytical model below breakpoint
			psi_sa_h:	coefficient for semianalytical model above breakpoint	
			c_sa_h:		coefficient for semianalytical model above breakpoint
			c_sa_l:		coefficient for semianalytical model below breakpoint
	"""
	c_sa_mean = (nb-1)/nsp*c_sa_l + (nsp-(nb-1))/nsp*c_sa_h	
	psi_sa_mean = (nb**2*(psi_sa_l-psi_sa_h)+nb*(psi_sa_h-psi_sa_l)+psi_sa_h*(nsp**2+nsp))/(nsp**2+nsp)

	return c_sa_mean, psi_sa_mean



def opt_nper_iter(nper,param):
	""" Calculate optimal number of periods for hybrid model iterativly
	inputs:	nt: total number of timesteps
			nb = breakpoint
			psi_sa: coefficient for semi analytical model
			c_sa:	coefficient for semi analytical model
			psi_FFT:coefficient for FFT model
	"""
	nt,nb,psi_sa_l,psi_sa_h,c_sa_l,c_sa_h,psi_FFT = param
	
	if nt/nper > nb:
		return np.sqrt(mean_params(nt/nper,nb,psi_sa_l,psi_sa_h,c_sa_l,c_sa_h)[1]*nt+mean_params(nt/nper,nb,psi_sa_l,psi_sa_h,c_sa_l,c_sa_h)[0]-2*psi_FFT)/np.sqrt(psi_FFT)-nper
	else:
		return np.sqrt(psi_sa_l*nt+c_sa_l-2*psi_FFT)/np.sqrt(psi_FFT)-nper
