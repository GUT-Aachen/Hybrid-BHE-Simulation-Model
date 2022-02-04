from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import sys
import time as time
from multiprocessing import Pool

sourcepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sourcepath)

import bhe_models 
import gfunctions  
import utilities 
import hybrid_model



def main():


	BheData = 		{	    'id'    : 'BHE01',
							'type' : '2-U',
							'length': 100.,			# [m]		length
							'diamB':  0.152,		# [m]		borehole diameter
							'lmG':    2.0,			# [W/m/K]	thermal conductivity grout
							'capG':   1000000,		# [MJ/m³/K]	volumetric heat capacity grout
							'lmP':    0.3,			# [W/m/K]	thermal conductivity pipe
							'odiamP': 0.032,		# [m]		outer diameter pipe
							'thickP': 0.0029,		# [m]		wall thickness pipe
							'distP':  0.04,			# [m]		pipe distance 
							'densF':  1045,			# [kg/m³]	density fluid	
							'dynviscF': 0.0035,		# [m]		dynamic viscosity fluid
							'lmF':    0.43,			# [m]		thermal conductivity fluid
							'capF':   3800000.,		# [m]		volumetric heat capacity fluid
							'covH':   1.,			# [m]		cover height
							'linkedHP': 'HP01',		# [m]		linked heat pump
							'xCoord' : 30000,		# [m]		x coordinate
							'yCoord' : 40000,		# [m]		y coordinate
							'Qf' : 1.92/3600.,		# [m³/s]	flow rate		
							'Tundist': 12.0,		# [°C]		undisturbed ground temperature
							'lm' : 2.3,				# [W/m/K]	thermal conductivity ground	
							'Cm' : 2300000.,		# [MJ/m³/K]	volumetric heat capacity ground
	}

	
	sim_setup = {	'nz': 	5,				# bhe cells vertical
				'dt_bhe': 	30,				# bhe time step
				'dt_soil': 	3600,			# soil time step
				'nt_part':	24,				# soil time steps per period
				'n_parts':	1,				# number of periods
				'error':	0.001,
				'type' : 'forward'			# euler scheme
	}	
	
	
	
	#create random Tin and flow rate
	M_Tin = np.ones(sim_setup['nt_part']*sim_setup['n_parts'])*50
	M_qf = np.ones(sim_setup['nt_part']*sim_setup['n_parts'])*BheData['Qf']


	x_pos = np.array([1])
	y_pos = np.array([1])
	nBhe = x_pos.size
	
	# -------------------------------------------------------------------------
	# calc gfunction and create time array tlog
	# -------------------------------------------------------------------------

	tstart = sim_setup['dt_soil']
	tstop = sim_setup['dt_soil']*sim_setup['nt_part']*sim_setup['n_parts']+1
	ntgfunc = 200
	
	tlog = np.geomspace(tstart, tstop, ntgfunc)
	gfuncs = gfunctions.g_Matrix(x_pos,y_pos,tlog,BheData)
	
	
	gfunc_data = { 'gfuncs' : gfuncs,
				   'gtime'  : tlog
				   }
	
	# gfunc for semi-analytical model
	tlin = np.linspace(sim_setup['dt_soil'],sim_setup['dt_soil']*sim_setup['nt_part'],sim_setup['nt_part'])	# time array soil
	gMatrix = np.zeros([nBhe,nBhe,sim_setup['nt_part']])										# matrix with gfuncs
	for i in range(0,nBhe):
		for j in range(0,nBhe):
			fG = interpolate.interp1d(gfunc_data['gtime'],gfunc_data['gfuncs'][i,j])
			gMatrix[i,j,:] = fG(tlin)																# gfunc matrix interpolated to linear time grid	
	
	# -------------------------------------------------------------------------
	# setup for simulation
	# -------------------------------------------------------------------------

	res_loads = np.zeros([nBhe,sim_setup['nt_part']*sim_setup['n_parts']])
	res_Tins = np.zeros([nBhe,sim_setup['nt_part']*sim_setup['n_parts']]) 
	res_Touts = np.zeros([nBhe,sim_setup['nt_part']*sim_setup['n_parts']])

	M_Tin = np.array([M_Tin])
	M_qf = np.array([M_qf])
	
	T_borehole = np.ones([nBhe,sim_setup['nt_part']])*BheData['Tundist']						
	T_borehole_ini = np.ones(nBhe)*BheData['Tundist']		
	
	BHEs = hybrid_model.init_BHEs(nBhe ,BheData,sim_setup['dt_bhe'],sim_setup['nz'], sim_setup['type'])

	# num first part
	start = time.time()
	(res_Tins[:,0:sim_setup['nt_part']],
	res_Touts[:,0:sim_setup['nt_part']],
	res_loads[:,0:sim_setup['nt_part']]) = hybrid_model.calc_sa_sec_U_forw(sim_setup,
																  gMatrix,
																  M_Tin[:,0:sim_setup['nt_part']],
																  M_qf[:,0:sim_setup['nt_part']],
																  T_borehole,BHEs,T_borehole_ini)
	
	for p in range(1,sim_setup['n_parts']):
			
		# analytical firt part
		T_borehole  = hybrid_model.calc_FFT_sec(gfunc_data['gfuncs'],
									 gfunc_data['gtime'],
									 res_loads[:,0:p*sim_setup['nt_part']],
									 sim_setup['nt_part'],
									 sim_setup['dt_soil'],
									 nBhe,BheData['Tundist'])
		
		T_borehole_ini = [T_borehole[k,p*sim_setup['nt_part']-1]  for k in range(0,nBhe)]		
		
		# num second part
		(res_Tins[:,p*sim_setup['nt_part']:(p+1)*sim_setup['nt_part']],
		res_Touts[:,p*sim_setup['nt_part']:(p+1)*sim_setup['nt_part']],
		res_loads[:,p*sim_setup['nt_part']:(p+1)*sim_setup['nt_part']]) = hybrid_model.calc_sa_sec_U_forw(sim_setup,
																								 gMatrix,
																								 M_Tin[:,p*sim_setup['nt_part']:(p+1)*sim_setup['nt_part']],
																								 M_qf[:,p*sim_setup['nt_part']:(p+1)*sim_setup['nt_part']],
																								 T_borehole[:,p*sim_setup['nt_part']:],BHEs,T_borehole_ini)
	
	print('time: ',time.time() - start)


	
	# -------------------------------------------------------------------------
	# plot results
	# -------------------------------------------------------------------------
	
	plt.plot(M_Tin[0],label = 'Tin')
	plt.plot(res_Touts[0],label = 'Tout')
	plt.legend()
	plt.show()






# Main function
if __name__ == '__main__':
    main()
