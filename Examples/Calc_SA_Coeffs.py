from __future__ import division

import numpy as np
import math

import matplotlib.pyplot as plt
import timeit
from scipy.optimize import curve_fit
import os
import sys

sourcepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sourcepath)

import utilities 


	
def nsec(nt,psi_FFT,psi_sa):
	return np.sqrt(psi_sa/psi_FFT*nt-1)


def calc_sa_coeff(T_borehole,loads,gMatrix,i,nSteps_soil,nBhe):
	for j in range(0,nBhe):
		for k in range(0,nBhe):	
			T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]

def main():

	nt = 250000
	stepsize = 1000
	nBhe = 1
	nb = 130000
	
	# random values for testing
	T_borehole = np.random.rand(nBhe,nt)
	gMatrix = np.random.rand(nBhe,nBhe,nt)
	loads = np.random.rand(nBhe,nt)	
	times_arr = np.zeros(int(nt/stepsize)-1)
	
	for j in range(stepsize,nt,stepsize):
		# measure computational times
		times_vec = timeit.repeat(lambda: calc_sa_coeff(T_borehole,loads,gMatrix,j,nt,nBhe),repeat =50, number = 10,globals=globals())	
		times_arr[int(j/stepsize)-1] = np.min(times_vec)/10		

	times_arr = np.flip(times_arr)
	steps = np.arange(stepsize,nt,stepsize)
	
	# curve fitting 
	# fit nr<nb
	popt_l, pcov = curve_fit(utilities.func_lin,steps[steps<nb], times_arr[steps<nb])
	m_low = popt_l[0]
	n_low = popt_l[1]
	print('m_low:, n_low: %d',m_low,n_low)

	# fit nr>nb
	popt_l, pcov = curve_fit(utilities.func_lin,steps[steps>nb], times_arr[steps>nb])
	m_high = popt_l[0]
	n_high = popt_l[1]
	print('m_high:, n_high %d',m_high,n_high)
	
	# - - - - - - - - - - - - - - - - - - - - - - - 
	#				plot results
	# - - - - - - - - - - - - - - - - - - - - - - - 
	
	fig = plt.figure(figsize=utilities.cm2inch(10, 6))
	ax1 = fig.add_subplot(111)	
	plt.subplots_adjust(bottom=0.22, top=0.90, left=0.18, right = 0.94) 

	ax1.set_ylabel(r'$\mathrm{Computation \: time \: [} 10^{-3} \: \mathrm{s]}$')
	ax1.set_xlabel(r'$\mathrm{remaining \: Steps \: [-]}$')

	ax1.plot(steps,1000*times_arr,label = r'$\mathrm{measured}$',color = 'k',linewidth = 0.7)

	ax1.plot(steps[steps<nb],1000*(steps[steps<nb]*m_low + n_low),label = r'$\mathrm{linear \: fit}$',linestyle = '--',color = 'grey',linewidth = 0.7,marker = 'o',markersize = 4,markevery=2,markerfacecolor='none')
	ax1.plot(steps[steps>nb],1000*(steps[steps>nb]*m_high + n_high),linestyle = '--',color = 'grey',linewidth = 0.7,marker = 'o',markersize = 4,markevery=2,markerfacecolor='none')

	ax1.legend(loc=0,frameon=False)
	plt.show()
	
	



# Main function
if __name__ == '__main__':
    main()