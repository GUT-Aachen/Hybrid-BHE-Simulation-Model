from __future__ import division
import os
import sys

sourcepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sourcepath)

import utilities 


nb = 130000		# breakpoint calc time sa model
nt = 2109906	# total number of timesteps

# computational time coefficients derived from measurements
psi_FFT = 1.58e-6		# coeff for FFT model
psi_sa_l = 5.8e-10		# coeff for sa model below breakpoint
psi_sa_h = 3.7e-9		# coeff for sa model above breakpoint
c_sa_l = 5.4e-6			# coeff for sa model below breakpoint
c_sa_h = -9.1e-5		# coeff for sa model above breakpoint


print(utilities.opt_nper(nt,psi_sa_l,psi_FFT,c_sa_l))

