from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import timeit

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d as interp1d

sourcepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sourcepath)

import utilities 


def calc_FFT_coeff(nt,gfunc,tlog,lin_gm,nBhe,nt_future,loads,dt):
	
	nsteps_total = loads.size + nt_future
	Time = np.linspace(dt,nsteps_total*dt,nsteps_total)

	# convert loads
	FFT_Loads = []
	for i in range(0,nBhe):
		gload = np.concatenate([loads,np.zeros(nt_future)])		
		gload[1:] -= np.roll(gload,1)[1:]
		
		FFT_Loads.append(utilities.FourierSci(gload))
	
	# Temperatures at each borehole
	T_borehole = np.ones([nBhe,Time.size])*10
	for i in range(0,nBhe):			
		for j in range(0,nBhe):
				
			# FFT Gfunc 
			fG = interp1d(tlog,gfunc)
			FFT_Gfunc = utilities.FourierSci(fG(Time))		
		
			# Calc Tborehole
			T_borehole[j,:] -= np.real(utilities.invFourierSci(FFT_Loads[i]*FFT_Gfunc)) 
	
	return True

	
def main():
	
	dt = 30
	nmax = 250000		
	stepsize = 7500	
	
	steps = np.arange(stepsize,nmax,stepsize)
	gfunc = np.random.rand(200)*10	
	times_arr = np.zeros(steps.size)

	
	for i in range(0,steps.size):
		print(steps[i])
		nt = steps[i]

		tlog = np.geomspace(dt,3*nt*dt,200)
		lin_g = interp1d(tlog,gfunc)
		nBhe = 1	
		loads = np.random.rand(nt)
		nt_future = loads.size
		
		times_arr[i] = np.min(timeit.repeat(lambda :calc_FFT_coeff(nt,gfunc,tlog,lin_g,nBhe,nt_future,loads,dt),repeat = 20, number = 1,globals=globals()))/1

	
	print(steps)
	popt_l, pcov = curve_fit(utilities.func_lin0,steps, times_arr)	
	
	print('m: '+ str(popt_l[0]))
	
	plt.plot(steps,times_arr,label = 'measured',color = 'b')
	plt.plot(steps,popt_l[0]*steps,label = 'lin fit', color = 'r')

	plt.ylabel('comp time')
	plt.xlabel('steps')
	plt.legend()
	plt.show()
	

# Main function
if __name__ == '__main__':
    main()