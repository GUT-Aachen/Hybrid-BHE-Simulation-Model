from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from multiprocessing import Pool
import os
import sys


import bhe_models 
import gfunctions  
import utilities 



def init_BHEs(nBhe,BheData,dt,nz,type):
	'''
	creates a list with initialized BHE classes
	inputs:		nBhe = number of BHEs to be initialized
				BheData = dict with BHE properties
				dt = time step
				nz = number of vertical cells for each BHE
				type = forward or backward euler model
	'''
	if type == 'backward':
		if BheData['type'] == '2-U':
			BHEs = [bhe_models.BHE_2U_impl() for i in range(nBhe)]
		if BheData['type'] == '1-U':
			BHEs = [bhe_models.BHE_1U_impl() for i in range(nBhe)]
		if BheData['type'] == 'Coax':
			BHEs = [bhe_models.BHE_Coax_impl() for i in range(nBhe)]
	
	if type == 'forward':
		if BheData['type'] == '2-U':
			BHEs = [bhe_models.BHE_2U_expl() for i in range(nBhe)]
		if BheData['type'] == '1-U':
			BHEs = [bhe_models.BHE_1U_expl() for i in range(nBhe)]
		if BheData['type'] == 'Coax':
			BHEs = [bhe_models.BHE_Coax_expl() for i in range(nBhe)]
	
	for i in range(nBhe):		
		BHEs[i].setTimestep(dt)
		BHEs[i].setnz(nz)
		BHEs[i].initialize(BheData)	
	return BHEs


def calc_sa_sec_U_forw(sim_setup,gMatrix,T_ins,M_qf,T_borehole,BHEs,T_borehole_ini):
	
	'''
	semi-analytical model for forward euler U-type BHE	
	inputs: sim_setup = dict with setup information
			gMatrix = Array with gfunctions
			Tins = List with inlet temperatures for each bhe for each time step [°C]
			M_qf = List with flow rates for each bhe for each time step [m³/s]
			T_borehole = Array with ground temperatures for each BHE
			BHEs = list with BHE classes
			T_borehole_ini = Array with initial groundtemperatures for the first time step
	'''
	
	
	nBhe = len(BHEs)												# number of bhe
	nSteps_soil = sim_setup['nt_part']								# soil steps per part
	Sondensteps = int(sim_setup['dt_soil']/sim_setup['dt_bhe'])		# bhe steps per soil step
	bhe_steps_inv = 1./Sondensteps

	
	# load Setup
	loads = [np.zeros(nSteps_soil) for i in range(nBhe)]	

	# Results Setup	
	Tf_outs = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_ins = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tg_means = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_out_mean = np.zeros(nSteps_soil)
	
	# temporary variables 
	Tf_in_ini = [np.zeros(BHEs[i].Tf_in.size) for i in range(nBhe)]
	Tf_out_ini = [np.zeros(BHEs[i].Tf_out.size) for i in range(nBhe)]
	Tg_in_ini = [np.zeros(BHEs[i].T_grout_in.size) for i in range(nBhe)]
	Tg_out_ini = [np.zeros(BHEs[i].T_grout_out.size) for i in range(nBhe)]
	
	# -------------------------------------------------------------------------
	# first step
	# -------------------------------------------------------------------------

	i = 0
	for m in range(0,nBhe):

		# save current state of bhe model
		Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
		Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
		Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
		Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]
	
		
		# First Guess	
		# Calc BHE numerical
		BHEs[m].setSoilBC(T_borehole_ini[m])	
		Tg_m = 0
		for n in range(0,Sondensteps):		

			if M_qf[m][i] > 0:
				BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
			else:
				BHEs[m].calcSondeNoFlow(1)	
			
			
			Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
		
		Tf_outs[m][i] = BHEs[m].getFluidOut()	
		load_fg = (T_borehole_ini[m]-Tg_m)*BHEs[m].Rgs_coupling
		Tb_guess = T_borehole[m][i] - (load_fg-0) * gMatrix[m][m][0]	
		loads[m][i] = load_fg
		
		error = np.inf
		while error > sim_setup['error']:
		
			# Sonde zurücksetzen!
			BHEs[m].alt_T_grout_in[:]  = Tg_in_ini[m][:]
			BHEs[m].alt_T_grout_out[:] = Tg_out_ini[m][:]
			BHEs[m].alt_Tf_in[:] = Tf_in_ini[m][:]
			BHEs[m].alt_Tf_out[:] = Tf_out_ini[m][:]
			
			
			# Calc BHE numerical
			Tf_out_old = BHEs[m].getFluidOut()	
			BHEs[m].setSoilBC(Tb_guess)
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if M_qf[m][i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)
					
				
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()
			load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
			loads[m][i] = load_fg
			Tb_guess = T_borehole[m][i] - (loads[m][i]-0) * gMatrix[m][m][0]	
			error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)


	
	# Calc Soil 
	for j in range(0,nBhe):		
		for k in range(0,nBhe):				
			T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]
	
	#print('step ' + str(i) + ' von ' + str(nSteps_soil))
	

	
	
	# -------------------------------------------------------------------------
	# following steps
	# -------------------------------------------------------------------------
	
	for i in range(1,nSteps_soil):
	
		for m in range(0,nBhe):
			# save current state of bhe model
			Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
			Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
			Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
			Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]
		
			

			# First Guess	
			# Calc BHE numerical
			BHEs[m].setSoilBC(T_borehole[m][i-1])																	
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if M_qf[m][i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)
								
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()	
			load_fg = (T_borehole[m][i-1]-Tg_m)*BHEs[m].Rgs_coupling
			Tb_guess = T_borehole[m][i] - (load_fg-loads[m][i-1]) * gMatrix[m][m][0]
			loads[m][i] = load_fg
			
			error = np.inf
			while error > sim_setup['error']:
			
				# Sonde zurücksetzen!
				BHEs[m].alt_T_grout_in[:]  = Tg_in_ini[m][:]
				BHEs[m].alt_T_grout_out[:] = Tg_out_ini[m][:]
				BHEs[m].alt_Tf_in[:] = Tf_in_ini[m][:]
				BHEs[m].alt_Tf_out[:] = Tf_out_ini[m][:]
				
				
				# Calc BHE numerical
				Tf_out_old = BHEs[m].getFluidOut()	
				BHEs[m].setSoilBC(Tb_guess)
				Tg_m = 0
				for n in range(0,Sondensteps):
					
					if M_qf[m][i] > 0:
						BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
					else:
						BHEs[m].calcSondeNoFlow(1)
						
					
					Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
				
				Tf_outs[m][i] = BHEs[m].getFluidOut()
				load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
				loads[m][i] = load_fg
				Tb_guess = T_borehole[m][i] - (loads[m][i]-loads[m][i-1]) * gMatrix[m][m][0]	
				error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)
		

		# Calc Soil 
		for j in range(0,nBhe):		
			for k in range(0,nBhe):				
				T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]

				
		#print('step ' + str(i) + ' von ' + str(nSteps_soil))	
		
		
	for m in range(0,nBhe):
		T_borehole_ini[m] = T_borehole[m][-1]

	return Tf_ins,Tf_outs,loads


def calc_sa_sec_U_backw(sim_setup,gMatrix,T_ins,M_qf,T_borehole,BHEs,T_borehole_ini):
	
	'''
	semi-analytical model for backward euler U-type BHE	
	inputs: sim_setup = dict with setup information
			gMatrix = Array with gfunctions
			Tins = List with inlet temperatures for each bhe for each time step [°C]
			M_qf = List with flow rates for each bhe for each time step [m³/s]
			T_borehole = Array with ground temperatures for each BHE
			BHEs = list with BHE classes
			T_borehole_ini = Array with initial groundtemperatures for the first time step
	'''
	

	nBhe = len(BHEs)												# number of bhe
	nSteps_soil = sim_setup['nt_part']								# soil steps per part
	Sondensteps = int(sim_setup['dt_soil']/sim_setup['dt_bhe'])		# bhe steps per soil step
	bhe_steps_inv = 1./Sondensteps

	
	# load Setup
	loads = [np.zeros(nSteps_soil) for i in range(nBhe)]	

	# Results Setup	
	Tf_outs = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_ins = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tg_means = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_out_mean = np.zeros(nSteps_soil)
	
	# temporary variables 
	Tf_in_ini = [np.zeros(BHEs[i].Tf_in.size) for i in range(nBhe)]
	Tf_out_ini = [np.zeros(BHEs[i].Tf_out.size) for i in range(nBhe)]
	Tg_in_ini = [np.zeros(BHEs[i].T_grout_in.size) for i in range(nBhe)]
	Tg_out_ini = [np.zeros(BHEs[i].T_grout_out.size) for i in range(nBhe)]
	
	# -------------------------------------------------------------------------
	# first step
	# -------------------------------------------------------------------------

	i = 0
	for m in range(0,nBhe):

		# save current state of bhe model
		Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
		Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
		Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
		Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]
	
		
		# First Guess	
		# Calc BHE numerical
		BHEs[m].setSoilBC(T_borehole_ini[m])	
		Tg_m = 0
		for n in range(0,Sondensteps):		
			
			if M_qf[m][i] > 0:
				BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
			else:
				BHEs[m].calcSondeNoFlow(1)	
			
			
			Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
		
		Tf_outs[m][i] = BHEs[m].getFluidOut()	
		load_fg = (T_borehole_ini[m]-Tg_m)*BHEs[m].Rgs_coupling
		Tb_guess = T_borehole[m][i] - (load_fg-0) * gMatrix[m][m][0]	
		loads[m][i] = load_fg
		
		error = np.inf
		while error > sim_setup['error']:
		
			# Reset BHE Model!
			BHEs[m].result[0:BHEs[m].nz]  = Tg_in_ini[m][:]
			BHEs[m].result[BHEs[m].nz:2*BHEs[m].nz] = Tg_out_ini[m][:]
			BHEs[m].result[2*BHEs[m].nz:3*BHEs[m].nz] = Tf_in_ini[m][:]
			BHEs[m].result[3*BHEs[m].nz:4*BHEs[m].nz] = Tf_out_ini[m][:]
			
			
			# Calc BHE numerical
			Tf_out_old = BHEs[m].getFluidOut()	
			BHEs[m].setSoilBC(Tb_guess)
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if M_qf[m][i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)
					
				
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()
			load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
			loads[m][i] = load_fg
			Tb_guess = T_borehole[m][i] - (loads[m][i]-0) * gMatrix[m][m][0]	
			error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)


	
	# Calc Soil 
	for j in range(0,nBhe):		
		for k in range(0,nBhe):				
			T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]
	
	#print('step ' + str(i) + ' von ' + str(nSteps_soil))
	

	
	
	# -------------------------------------------------------------------------
	# following steps
	# -------------------------------------------------------------------------
	
	for i in range(1,nSteps_soil):
	
		for m in range(0,nBhe):
			# save current state of bhe model
			Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
			Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
			Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
			Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]
		
			

			# First Guess	
			# Calc BHE numerical
			BHEs[m].setSoilBC(T_borehole[m][i-1])																	
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if M_qf[m][i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)		
				
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()	
			load_fg = (T_borehole[m][i-1]-Tg_m)*BHEs[m].Rgs_coupling
			Tb_guess = T_borehole[m][i] - (load_fg-loads[m][i-1]) * gMatrix[m][m][0]
			loads[m][i] = load_fg
			
			error = np.inf
			while error > sim_setup['error']:
			
				# Reset BHE Model!
				BHEs[m].result[0:BHEs[m].nz]  = Tg_in_ini[m][:]
				BHEs[m].result[BHEs[m].nz:2*BHEs[m].nz] = Tg_out_ini[m][:]
				BHEs[m].result[2*BHEs[m].nz:3*BHEs[m].nz] = Tf_in_ini[m][:]
				BHEs[m].result[3*BHEs[m].nz:4*BHEs[m].nz] = Tf_out_ini[m][:]
				
				
				# Calc BHE numerical
				Tf_out_old = BHEs[m].getFluidOut()	
				BHEs[m].setSoilBC(Tb_guess)
				Tg_m = 0
				for n in range(0,Sondensteps):
					
					if M_qf[m][i] > 0:
						BHEs[m].calcSondeFlowQ(1,T_ins[m][i],M_qf[m][i])
					else:
						BHEs[m].calcSondeNoFlow(1)
						
					
					Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
					
				Tf_outs[m][i] = BHEs[m].getFluidOut()	
				load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
				loads[m][i] = load_fg
				Tb_guess = T_borehole[m][i] - (loads[m][i]-loads[m][i-1]) * gMatrix[m][m][0]	
				error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)

		# Calc Soil 
		for j in range(0,nBhe):		
			for k in range(0,nBhe):				
				T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]

				
		#print('step ' + str(i) + ' von ' + str(nSteps_soil))	
		
		
		
	for m in range(0,nBhe):
		T_borehole_ini[m] = T_borehole[m][-1]


	return Tf_ins,Tf_outs,loads


def calc_sa_sec_U_backw_load(sim_setup,gMatrix,M_qf,field_load,T_borehole,BHEs,Tmean_out_old):

	'''
	semi-analytical model for backward euler U-type BHE	with load as BC
	- all BHE have the same inlet temperatures
	- inlet temperatures are derived from load and mean outlet temperatures
	inputs: sim_setup = dict with setup information
			gMatrix = Array with gfunctions
			field_load = total load for all BHE combined
			M_qf = List with flow rates for each bhe for each time step [m³/s]
			T_borehole = Array with ground temperatures for each BHE
			BHEs = list with BHE classes
			Tmean_out_old = mean outlet temperature of all BHE of last period
	'''
	

	nBhe = len(BHEs)												# number of bhe
	nSteps_soil = sim_setup['nt_part']								# soil steps per part
	Sondensteps = int(sim_setup['dt_soil']/sim_setup['dt_bhe'])		# bhe steps per soil step
	bhe_steps_inv = 1./Sondensteps
	nBhe_inv = 1./nBhe

	
	# load Setup
	loads = [np.zeros(nSteps_soil) for i in range(nBhe)]	
	pcV_inv = 1/(nBhe*BHEs[0].Qold*BHEs[0].capF)

	# Results Setup	
	Tf_outs = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_ins = np.zeros(nSteps_soil)
	Tg_means = [np.zeros(nSteps_soil) for i in range(nBhe)]
	Tf_out_mean = np.zeros(nSteps_soil)
	
	# temporary variables 
	Tf_in_ini = [np.zeros(BHEs[i].Tf_in.size) for i in range(nBhe)]
	Tf_out_ini = [np.zeros(BHEs[i].Tf_out.size) for i in range(nBhe)]
	Tg_in_ini = [np.zeros(BHEs[i].T_grout_in.size) for i in range(nBhe)]
	Tg_out_ini = [np.zeros(BHEs[i].T_grout_out.size) for i in range(nBhe)]
	
	# -------------------------------------------------------------------------
	# first step
	# -------------------------------------------------------------------------
	i = 0
	T_in = Tmean_out_old - field_load[i]*pcV_inv	
	Tf_ins[i] = T_in
	for m in range(0,nBhe):
	
		# save current state of bhe model
		Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
		Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
		Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
		Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]
	
		
		# First Guess	
		# Calc BHE numerical
		BHEs[m].setSoilBC(T_borehole[m][0])	
		Tg_m = 0
		for n in range(0,Sondensteps):		
			
			if field_load[i] > 0:
				BHEs[m].calcSondeFlowQ(1,T_in,M_qf[m][i])
			else:
				BHEs[m].calcSondeNoFlow(1)	
			
			
			Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
		
		Tf_outs[m][i] = BHEs[m].getFluidOut()	
		load_fg = (T_borehole[m][0]-Tg_m)*BHEs[m].Rgs_coupling
		Tb_guess = T_borehole[m][i] - (load_fg-0) * gMatrix[m][m][0]	
		loads[m][i] = load_fg
		
		error = np.inf
		while error > sim_setup['error']:
		
			# Reset BHE Model!
			BHEs[m].result[0:BHEs[m].nz]  = Tg_in_ini[m][:]
			BHEs[m].result[BHEs[m].nz:2*BHEs[m].nz] = Tg_out_ini[m][:]
			BHEs[m].result[2*BHEs[m].nz:3*BHEs[m].nz] = Tf_in_ini[m][:]
			BHEs[m].result[3*BHEs[m].nz:4*BHEs[m].nz] = Tf_out_ini[m][:]
			
			
			# Calc BHE numerical
			Tf_out_old = BHEs[m].getFluidOut()	
			BHEs[m].setSoilBC(Tb_guess)
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if field_load[i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_in,M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)
					
				
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()		
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()
			load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
			loads[m][i] = load_fg
			Tb_guess = T_borehole[m][i] - (loads[m][i]-0) * gMatrix[m][m][0]	
			error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)
		
		Tf_out_mean[i] += nBhe_inv * Tf_outs[m][i]

	
	# Calc Soil 
	for j in range(0,nBhe):		
		for k in range(0,nBhe):				
			T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]
	
	#print('step ' + str(i) + ' von ' + str(nSteps_soil))
	

	
	
	# -------------------------------------------------------------------------
	# following steps
	# -------------------------------------------------------------------------
	
	for i in range(1,nSteps_soil):
		T_in = Tf_out_mean[i-1] - field_load[i]*pcV_inv
		Tf_ins[i] = T_in
		for m in range(0,nBhe):
			# save current state of bhe model
			Tf_in_ini[m][:] = BHEs[m].Tf_in[:]
			Tf_out_ini[m][:] = BHEs[m].Tf_out[:]
			Tg_in_ini[m][:] = BHEs[m].T_grout_in[:]
			Tg_out_ini[m][:] = BHEs[m].T_grout_out[:]

			# First Guess	
			# Calc BHE numerical
			BHEs[m].setSoilBC(T_borehole[m][i-1])																	
			Tg_m = 0
			for n in range(0,Sondensteps):
				
				if M_qf[m][i] > 0:
					BHEs[m].calcSondeFlowQ(1,T_in,M_qf[m][i])
				else:
					BHEs[m].calcSondeNoFlow(1)
							
				
				Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
			
			Tf_outs[m][i] = BHEs[m].getFluidOut()	
			load_fg = (T_borehole[m][i-1]-Tg_m)*BHEs[m].Rgs_coupling
			Tb_guess = T_borehole[m][i] - (load_fg-loads[m][i-1]) * gMatrix[m][m][0]
			loads[m][i] = load_fg
			
			error = np.inf
			while error > sim_setup['error']:
			
				# Reset BHE Model!
				BHEs[m].result[0:BHEs[m].nz]  = Tg_in_ini[m][:]
				BHEs[m].result[BHEs[m].nz:2*BHEs[m].nz] = Tg_out_ini[m][:]
				BHEs[m].result[2*BHEs[m].nz:3*BHEs[m].nz] = Tf_in_ini[m][:]
				BHEs[m].result[3*BHEs[m].nz:4*BHEs[m].nz] = Tf_out_ini[m][:]
				
				
				# Calc BHE numerical
				Tf_out_old = BHEs[m].getFluidOut()	
				BHEs[m].setSoilBC(Tb_guess)
				Tg_m = 0
				for n in range(0,Sondensteps):
					
					if M_qf[m][i] > 0:
						BHEs[m].calcSondeFlowQ(1,T_in,M_qf[m][i])
					else:
						BHEs[m].calcSondeNoFlow(1)
						
					
					Tg_m += bhe_steps_inv * BHEs[m].getGroutBC()			
					
				Tf_outs[m][i] = BHEs[m].getFluidOut()	
				load_fg = (Tb_guess-Tg_m)*BHEs[m].Rgs_coupling
				loads[m][i] = load_fg
				Tb_guess = T_borehole[m][i] - (loads[m][i]-loads[m][i-1]) * gMatrix[m][m][0]	
				error = np.abs((BHEs[m].getFluidOut() - Tf_out_old)/Tf_out_old)
				
			Tf_out_mean[i] += nBhe_inv * Tf_outs[m][i]

		# Calc Soil 
		for j in range(0,nBhe):		
			for k in range(0,nBhe):				
				T_borehole[k][i:nSteps_soil] -= (loads[j][i]-loads[j][i-1]) * gMatrix[j,k,0:nSteps_soil-i]

				
		#print('step ' + str(i) + ' von ' + str(nSteps_soil))	
		
		
	

	#print ("Time NumSec: " +str(time.time() - start))
	return Tf_ins,Tf_outs,loads


def calc_FFT_sec(gfuncs,tlog,loads,nt_future,dt,nBhe,T_undist):
	
	'''
	FFT model for calculation of borehole temperatures for next period
	inputs:	gfuncs = array with gfunctions over log time
			tlog = corresponding time array for gfuncs
			loads = list with loads for eachs borehole
			nt_future = number of timesteps of the next period
			dt = timestep size
			nBhe = number of BHE
			T_undist = undisturbed ground temperature
	'''
	
	nsteps_total = loads[0].size + nt_future
	Time = np.linspace(dt,nsteps_total*dt,nsteps_total)

	# convert loads
	FFT_Loads = []
	for i in range(0,nBhe):
		gload = np.concatenate([loads[i],np.zeros(nt_future)])		
		gload[1:] -= np.roll(gload,1)[1:]
		
		FFT_Loads.append(utilities.FourierSci(gload))
	
	# Temperatures at each borehole
	T_borehole = np.ones([nBhe,Time.size])*T_undist
	for i in range(0,nBhe):			
		for j in range(0,nBhe):
				
			# FFT Gfunc 
			fG = interpolate.interp1d(tlog,gfuncs[i,j,:])
			FFT_Gfunc = utilities.FourierSci(fG(Time))		
		
			# Calc Tborehole
			T_borehole[j,:] -= np.real(utilities.invFourierSci(FFT_Loads[i]*FFT_Gfunc)) 
	
	return T_borehole


def calc_FFT_sec_parallel(gfuncs,tlog,loads,nt_future,dt,nBhe,T_undist):
	
	'''
	FFT model for calculation of borehole temperatures for next period
	- FFT is solved parallel for multiple boreholes
	inputs:	gfuncs = array with gfunctions over log time
			tlog = corresponding time array for gfuncs
			loads = list with loads for eachs borehole
			nt_future = number of timesteps of the next period
			dt = timestep size
			nBhe = number of BHE
			T_undist = undisturbed ground temperature
	'''
	
	nsteps_total = loads[0].size + nt_future
	Time = np.linspace(dt,nsteps_total*dt,nsteps_total)

	# convert loads
	FFT_Loads = []
	for i in range(0,nBhe):
		gload = np.concatenate([loads[i],np.zeros(nt_future)])
		gload[1:] -= np.roll(gload,1)[1:]		
		
		FFT_Loads.append(utilities.FourierSci(gload))
		
	# Temperatures at each borehole
	T_borehole = np.ones([nBhe,Time.size])*T_undist
	inputs = []
	pool = Pool(processes=None)
	for i in range(0,nBhe):				
		input = [tlog,gfuncs,nBhe,Time,FFT_Loads,i]
		inputs.append(input)
		
	res = pool.map(calc_Tb_parallel,inputs)
	
	for result in res:
		T_borehole[result[0],:] += result[1]
	
	pool.close()
	pool.join()
	
	return T_borehole

	
def calc_Tb_parallel(inputs):
	
	'''
	FFT model parallel (for large number of boreholes)
	gets called by calc_FFT_sec_parallel
	'''
	
	tlog = inputs[0]
	gfuncs = inputs[1]
	nBhe = inputs[2]
	Time = inputs[3]
	FFT_Loads = inputs[4]
	i = inputs[5]
	
	T_borehole = np.zeros(Time.size)
	for j in range(0,nBhe):
		# FFT Gfunc 
		fG = interpolate.interp1d(tlog,gfuncs[i,j,:])
		FFT_Gfunc = utilities.FourierSci(fG(Time))		
	
		# Calc Tborehole
		T_borehole -= np.real(utilities.invFourierSci(FFT_Loads[j]*FFT_Gfunc)) 
		
	return [i,T_borehole]

