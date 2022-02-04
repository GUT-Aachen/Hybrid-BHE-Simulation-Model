from __future__ import division

import numpy as np
import math

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar



class BHE_CXA_impl:

	'''
	implicit simulation model for coaxial bhe with anular inlet
	
	'''
	
	
	def setTimestep(self,dt):
		self.dt = dt
		
	def setnz(self,nz):
		self.nz = nz
	
	def setSoilBC(self,Tsoil):
		self.result[3*self.nz:4*self.nz] = Tsoil
	
	def getGroutBC(self):
		return np.mean(self.T_grout[1:])
		
	def getFluidOut(self):
		return self.Tf_out[1]
		
	def getFluidIn(self):
		return self.Tf_in[1]
		
	def initialize(self,BheData):		
		self.dynviscF = BheData['dynviscF']
		self.dz = BheData['length']/self.nz
		self.nz = self.nz+1
		
		# Geometry
		self.ro = BheData['diamB']/2
		self.ri_i = BheData['odiamPin']/2-BheData['thickPin']
		self.ri_o = BheData['odiamPin']/2
		self.ro_i = BheData['odiamPout']/2-BheData['thickPout']
		self.ro_o = BheData['odiamPout']/2
		self.di_i = BheData['odiamPin']-2*BheData['thickPin']
		self.do_i = BheData['odiamPout']-2*BheData['thickPout']
		self.dh = self.di_i-BheData['odiamPout']	
		
		# Flow velocity
		self.uo = BheData['Qf']/(np.pi*self.ro_i**2)
		self.ui = BheData['Qf']/(np.pi*(self.ri_i**2-self.ro_o**2))		
		self.maxVel = np.max([self.ui,self.uo]) 	
	
		# Flow parameters
		self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Reo = self.uo*self.do_i/(self.dynviscF/BheData['densF'])
		self.Rei = self.ui*self.dh/(self.dynviscF/BheData['densF'])
		self.Nuo = NusseltCoaxo (self.Reo,self.Pr,self.ro_i,BheData['length'])
		self.Nui = NusseltCoaxi (self.Rei,self.Pr,BheData['odiamPout'],self.di_i,self.dh,BheData['length'])	
		self.Radvo  = 1/(self.Nuo*BheData['lmF']*np.pi)
		self.Radvia = 1/(self.Nui*BheData['lmF']*np.pi) * self.dh/BheData['odiamPout']
		self.Radvib = 1/(self.Nui*BheData['lmF']*np.pi) * self.dh/self.di_i
		
		# Resistances
		self.x = np.log(((BheData['diamB']**2+BheData['odiamPin']**2)**0.5)/((2**0.5)*BheData['odiamPin']))/np.log(BheData['diamB']/BheData['odiamPin'])
		self.Rg = np.log(BheData['diamB']/BheData['odiamPin'])/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg
		self.Rgs_coupling = 1./self.Rgs
		self.Rconb = self.x*self.Rg
		self.Rcono = np.log(self.ro_o/self.ro_i)/(2*np.pi*BheData['lmPout'])
		self.Rconi = np.log(self.ri_o/self.ri_i)/(2*np.pi*BheData['lmPin'])
		self.Rff = self.Radvo + self.Radvia + self.Rcono
		self.Rfig = self.Radvib + self.Rconi + self.Rconb
		self.RffNoFlow = self.Rcono
		self.RfigNoFlow = self.Rconb + self.Rconi
		
				
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-self.ri_o**2)		# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ain = np.pi*(self.ri_i**2-self.ro_o**2)	# area pipeIn/fluidIn
		self.Vin = self.Ain*self.dz						# volume pipeIn/fluidIn
		self.Aout = np.pi*self.ro_i**2					# area pipeOut/fluidOut
		self.Vout = self.Aout*self.dz					# volume pipeOut/fluidOut
		
		# Variables Flow
		self.F1 = self.dz/self.Rgs						
		self.F2 = self.dz/self.Rfig			
		self.F3 = self.dt/self.Vg/BheData['capG']		
		self.F4 = self.dz/self.Rff
		self.F5i = self.dt/self.dz*self.ui				
		self.F5o = self.dt/self.dz*self.uo
		self.F6 = self.dt/self.Vin/BheData['capF']
		self.F7 = self.dt/self.Vout/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag				
		
		self.F6F2 = self.F6*self.F2						
		self.F6F3 = self.F6*self.F3
		self.F6F4 = self.F6*self.F4
		self.F7F4 = self.F7*self.F4
		self.F3F1 = self.F3*self.F1						
		self.F3F2 = self.F3*self.F2
		self.F3F8 = self.F3*self.F8						
	
		
		# Variables NoFlow
		self.F2NoFlow = self.dz/self.RfigNoFlow
		self.F4NoFlow = self.dz/self.RffNoFlow	
		self.F6F2NoFlow = self.F6*self.F2NoFlow
		self.F6F4NoFlow = self.F6*self.F4NoFlow
		self.F7F4NoFlow = self.F7*self.F4NoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow
		self.F5iNoFlow = 0				
		self.F5oNoFlow = 0

		# Set up matrix Flow
		# data = data array, posX = X Position, posY = Y Position
		self.na = 4 # number of cell arrays, Tin, Tout, Tgrout, Tsoil
		##### Grout
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2)
		self.dataTemp[0] = 1+self.F3F8+self.F3F1+self.F3F2 # oberste Zelle
		self.dataTemp[self.nz-1] = 1+self.F3F8+self.F3F1+self.F3F2 # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## grout z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## grout z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## grout to soil 
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data, self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## grout to tin 
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)

		## main diagonal Tin 
		self.dataTemp = np.ones(self.nz)*(1+self.F5i+self.F6F2+self.F6F4)
		self.dataTemp[0] = (1+self.F6F2+self.F6F4)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + self.nz)
		## Tin z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5i)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ self.nz)
		## Tin to grout 
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))
		## Tin to Tout 
		self.dataTemp = np.ones(self.nz)*-self.F6F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-2)*self.nz)

		## main diagonal Tout 
		self.data = np.append(self.data,np.ones(self.nz)*(1+self.F5o+self.F7F4))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tout z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5o)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+2*self.nz)
		## Tout to tin 
		self.dataTemp = np.ones(self.nz)*-self.F7F4
		self.dataTemp[self.nz-1] = -(self.F5o+self.F7F4)
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)

		## main diagonal Soil 
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)

		self.K_Flow = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_Flow = sparse.csr_matrix(self.K_Flow)
		
		
		# Set up matrix NoFlow
		# data = Datenarray, posX = X Position, posY = Y Position, dataTemp = Temporär
		self.na = 4 # number of cell arrays, Tin, Tout, Tgrout, Tsoil
		##### Grout
		## main diagonal grout 
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2NoFlow)
		self.dataTemp[0] = 1+self.F3F8+self.F3F1+self.F3F2NoFlow # oberste Zelle
		self.dataTemp[self.nz-1] = 1+self.F3F8+self.F3F1+self.F3F2NoFlow # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## grout z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## grout z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## grout to soil 
		self.data = np.append(self.data,np.ones(self.nz)*-self.F3F1)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## grout to tin
		self.data = np.append(self.data,np.ones(self.nz)*-self.F3F2NoFlow)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)

		## main diagonal Tin 
		self.dataTemp = np.ones(self.nz)*(1+self.F5iNoFlow+self.F6F2NoFlow+self.F6F4NoFlow)
		self.dataTemp[0] = (1+self.F6F2NoFlow+self.F6F4NoFlow)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + self.nz)
		## Tin z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5iNoFlow)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ self.nz)
		## Tin to grout 
		self.data = np.append(self.data,np.ones(self.nz)*-self.F6F2NoFlow)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))
		## Tin to Tout
		self.data = np.append(self.data,np.ones(self.nz)*-self.F6F4NoFlow)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-2)*self.nz)

		## main diagonal Tout 
		self.data = np.append(self.data,np.ones(self.nz)*(1+self.F5oNoFlow+self.F7F4NoFlow))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tout z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5oNoFlow)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+2*self.nz)
		## Tout to tin
		self.dataTemp = np.ones(self.nz)*-self.F7F4NoFlow
		self.dataTemp[self.nz-1] = -(self.F5oNoFlow+self.F7F4NoFlow)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)

		## main diagonal Soil
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)

		self.K_NoFlow = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_NoFlow = sparse.csr_matrix(self.K_NoFlow)
		
		######## Told ########
		######################
		self.result = np.ones(self.nz*self.na)*BheData['Tundist'] # Initialbedingung
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.T_grout = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	

		for i in range(0,tfinal):
			self.result[self.nz] = T_in	# BC Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Solve EQS
			# Rausschreiben für Plot
			self.T_grout[:] = self.U[0:self.nz]
			self.Tf_in[:] = self.U[self.nz:2*self.nz]
			self.Tf_out[:] = self.U[2*self.nz:3*self.nz]
			
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
									
		return True
	
	def calcSondeNoFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout
			self.result[self.nz] = T_in	# Vorgabe Randbedingung Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_NoFlow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout[:] = self.U[0:self.nz]
			self.Tf_in[:] = self.U[self.nz:2*self.nz]
			self.Tf_out[:] = self.U[2*self.nz:3*self.nz]
			
			
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 			
		return True
	
class BHE_CXA_expl:

	'''
	explicit simulation model for coaxial bhe with anular inlet
	
	'''
		
	def setTimestep(self,dt):
		self.dt = dt
		
	def setnz(self,nz):
		self.nz = nz
	
	def setSoilBC(self,Tsoil):
		self.TsA[:] = Tsoil
	
	def getGroutBC(self):
		return np.mean(self.T_grout)
		
	def getFluidOut(self):
		return self.Tf_out[0]
		
	def getFluidIn(self):
		return self.Tf_in[0]
		
	def initialize(self,BheData):		
		self.dz = BheData['length']/self.nz
		dynviscF = BheData['dynviscF']
		
		# Geometry
		D = BheData['diamB']
		self.ro = BheData['diamB']/2
		self.ri_i = BheData['odiamPin']/2-BheData['thickPin']
		self.ri_o = BheData['odiamPin']/2
		self.ro_i = BheData['odiamPout']/2-BheData['thickPout']
		self.ro_o = BheData['odiamPout']/2
		self.di_i = BheData['odiamPin']-2*BheData['thickPin']
		self.do_i = BheData['odiamPout']-2*BheData['thickPout']
		self.dh = self.di_i-BheData['odiamPout']	
		
		# Flow velocity
		self.uo = BheData['Qf']/(np.pi*self.ro_i**2)
		self.ui = BheData['Qf']/(np.pi*(self.ri_i**2-self.ro_o**2))		
		self.maxVel = np.max([self.ui,self.uo]) 	

	
		# Flow parameters
		self.Pr = BheData['dynviscF']*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Reo = self.uo*self.do_i/(BheData['dynviscF']/BheData['densF'])
		self.Rei = self.ui*self.dh/(BheData['dynviscF']/BheData['densF'])
		self.Nuo = NusseltCoaxo (self.Reo,self.Pr,self.ro_i,BheData['length'])
		self.Nui = NusseltCoaxi (self.Rei,self.Pr,BheData['odiamPout'],self.di_i,self.dh,BheData['length'])	
		self.Radvo  = 1/(self.Nuo*BheData['lmF']*np.pi)
		self.Radvia = 1/(self.Nui*BheData['lmF']*np.pi) * self.dh/BheData['odiamPout']
		self.Radvib = 1/(self.Nui*BheData['lmF']*np.pi) * self.dh/self.di_i
		
		# Resistances
		self.x = np.log(((BheData['diamB']**2+BheData['odiamPin']**2)**0.5)/((2**0.5)*BheData['odiamPin']))/np.log(BheData['diamB']/BheData['odiamPin'])
		self.Rg = np.log(BheData['diamB']/BheData['odiamPin'])/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg
		self.Rgs_coupling = 1./self.Rgs
		self.Rconb = self.x*self.Rg
		self.Rcono = np.log(self.ro_o/self.ro_i)/(2*np.pi*BheData['lmPout'])
		self.Rconi = np.log(self.ri_o/self.ri_i)/(2*np.pi*BheData['lmPin'])
		self.Rff = self.Radvo + self.Radvia + self.Rcono
		self.Rfig = self.Radvib + self.Rconi + self.Rconb
		self.RffNoFlow = self.Rcono
		self.RfigNoFlow = self.Rconb + self.Rconi
				
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-self.ri_o**2)		# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ain = np.pi*(self.ri_i**2-self.ro_o**2)	# area pipeIn/fluidIn
		self.Vin = self.Ain*self.dz						# volume pipeIn/fluidIn
		self.Aout = np.pi*self.ro_i**2					# area pipeOut/fluidOut
		self.Vout = self.Aout*self.dz					# volume pipeOut/fluidOut

		
		# Variables Flow
		self.F1 = self.dz/self.Rgs
		self.F2 = self.dz/self.Rfig
		self.F3 = self.dt/self.Vg/BheData['capG']
		self.F4 = self.dz/self.Rff
		self.F5i = self.dt/self.dz*self.ui
		self.F5o = self.dt/self.dz*self.uo
		self.F6 = self.dt/self.Vin/BheData['capF']
		self.F7 = self.dt/self.Vout/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag		
		
		self.F6F2 = self.F6*self.F2
		self.F6F3 = self.F6*self.F3
		self.F6F4 = self.F6*self.F4
		self.F7F4 = self.F7*self.F4
		self.F3F1 = self.F3*self.F1
		self.F3F2 = self.F3*self.F2
		self.F3F8 = self.F3*self.F8
		
		
		# Variables NoFlow
		self.F2NoFlow = self.dz/self.RfigNoFlow
		self.F4NoFlow = self.dz/self.RffNoFlow	
		self.F6F2NoFlow = self.F6*self.F2NoFlow
		self.F6F4NoFlow = self.F6*self.F4NoFlow
		self.F7F4NoFlow = self.F7*self.F4NoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow

		self.idxPlus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),1)
		self.idxMinus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),-1)

		self.Q_cond = np.zeros(self.nz)
		self.Phi_w = np.zeros(self.nz)
		self.Phi_e = np.zeros(self.nz)
		
		# Calc Arrays
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.T_grout = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		self.TsA = np.full(self.nz,BheData['Tundist'])  		# CellArray Average Soil Temperature
		self.zgrid = np.linspace(0,BheData['length'],self.nz)
		self.alt_Tf_in = np.full(self.nz,BheData['Tundist']) 	# CellArray Tf_in
		self.alt_Tf_out = np.full(self.nz,BheData['Tundist']) 	# CellArray Tf_out
		self.alt_T_grout = np.full(self.nz,BheData['Tundist'])  # CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout
			self.Q_cond = (np.take(self.alt_T_grout,self.idxMinus)-2*self.alt_T_grout+np.take(self.alt_T_grout,self.idxPlus))
			self.Q_cond[0] = (self.alt_T_grout[1]-self.alt_T_grout[0])
			self.Q_cond[self.nz-1] = (self.alt_T_grout[self.nz-2]-self.alt_T_grout[self.nz-1])
			self.T_grout = self.alt_T_grout + self.F3F8 * self.Q_cond + self.F3F1*(self.TsA-self.alt_T_grout) + self.F3F2*(self.alt_Tf_in-self.alt_T_grout)

			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Phi_w[0] = T_in
			self.Tf_in = self.alt_Tf_in + self.F5i*(self.Phi_w - self.alt_Tf_in) + self.F6F2*(self.alt_T_grout-self.alt_Tf_in) + self.F6F4*(self.alt_Tf_out-self.alt_Tf_in)
					
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_out,self.idxMinus)
			self.Phi_w[self.nz-1] = self.alt_Tf_in[self.nz-1]			
			self.Tf_out = self.alt_Tf_out + self.F5o * (self.Phi_w - self.alt_Tf_out) + self.F7F4*(self.alt_Tf_in-self.alt_Tf_out)
				
			# Rückschreiben
			self.alt_T_grout[:] = self.T_grout[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]
									
		return True
		
	def calcSondeNoFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout
			self.Q_cond = (np.take(self.alt_T_grout,self.idxMinus)-2*self.alt_T_grout+np.take(self.alt_T_grout,self.idxPlus))
			self.Q_cond[0] = (self.alt_T_grout[1]-self.alt_T_grout[0])
			self.Q_cond[self.nz-1] = (self.alt_T_grout[self.nz-2]-self.alt_T_grout[self.nz-1])
			self.T_grout = self.alt_T_grout + self.F3F8 * self.Q_cond + self.F3F1*(self.TsA-self.alt_T_grout) + self.F3F2NoFlow*(self.alt_Tf_in-self.alt_T_grout)
			
			# Berechnung fluid in
			self.Tf_in = self.alt_Tf_in + self.F6F2NoFlow*(self.alt_T_grout-self.alt_Tf_in) + self.F6F4NoFlow*(self.alt_Tf_out-self.alt_Tf_in)
					
			# Berechnung fluid out		
			self.Tf_out = self.alt_Tf_out + self.F7F4NoFlow*(self.alt_Tf_in-self.alt_Tf_out)
				
			# Rückschreiben
			self.alt_T_grout[:] = self.T_grout[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]			
		return True

class BHE_2U_impl:

	'''
	implicit simulation model for 2U bhe
	
	'''
	
	def setSoilBC(self,Tsoil):
		self.result[4*self.nz:5*self.nz] = Tsoil
	
	def getGroutBC(self):
		return (np.mean(self.T_grout_in[1:])+np.mean(self.T_grout_out[1:]))/2
	
	def getFluidOut(self):
		return self.Tf_out[1]
		
	def getFluidIn(self):
		return self.Tf_in[1]	

	def setTimestep(self,dt):
		self.dt = dt	
		
	def setnz(self,nz):
		self.nz = nz
		
	def initialize(self,BheData):		
		
		# discretize
		self.dz = BheData['length']/self.nz
		self.nz = self.nz+1
		
		# Geometry
		self.D = BheData['diamB']
		self.ro = BheData['diamB']/2
		self.rpi = BheData['odiamP']/2-BheData['thickP']
		self.rpo = BheData['odiamP']/2
		self.dpi = BheData['odiamP']-2*BheData['thickP']
		self.dpo = BheData['odiamP']
		
		# fluid properties
		self.dynviscF = BheData['dynviscF']
		self.densF = BheData['densF']
		self.lmF = BheData['lmF']
		self.length	= BheData['length']
		self.capF = BheData['capF']
		
		# Flow velocity
		self.u = BheData['Qf']/(2*np.pi*self.rpi**2)
		self.Qold = BheData['Qf']
		self.maxVel = self.u 
	
		# Flow parameters
		self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Re = self.u*self.dpi/(self.dynviscF/BheData['densF'])
		self.Nu = Nusselt2U (self.Re,self.Pr,self.rpi,BheData['length'])
		self.Radv  = 1/(self.Nu*BheData['lmF']*np.pi)
		
		# Thermal Resistances		
		self.s = BheData['distP']*2**0.5
		self.x = np.log((self.D**2+4*self.dpo**2)**0.5/(2*2**0.5*self.dpo))/np.log(self.D/(2*self.dpo))
		self.Rg = (3.098-4.432*self.s/self.D + 2.364*self.s**2/(self.D**2)) * np.arccosh((self.D**2+self.dpo**2-self.s**2)/(2*self.D*self.dpo))/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg
		self.Rconb = self.x*self.Rg
		self.Rcona = np.log(self.rpo/self.rpi)/(2*np.pi*BheData['lmP'])
		self.Rar1 = np.arccosh((self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])
		self.Rar2 = np.arccosh((2*self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])
		self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
		
		### Check negative thermal resistances ###
		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):
			self.x = 2/3*self.x	
			self.Rconb = self.x*self.Rg
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
					
		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):
			self.x = self.x * 1/3	
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)

		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):
			self.x = 0
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
		
		self.Rgs_coupling = 4./self.Rgs
		self.Rfg = self.Radv + self.Rcona + self.Rconb	
		self.RfgNoFlow = self.Rconb + self.Rcona
		
				
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-4*self.rpo**2)/4	# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ap = np.pi*self.rpi**2						# area pipe/fluid
		self.Vp = self.Ap*self.dz						# volume pipe/fluid
		
		# Variables Flow					# Ab hier werden werte zusammengefasst damit rechnung schneller läuft
		self.F1 = self.dz/self.Rgs			# am besten zurücksubstituieren für verständnis
		self.F2 = self.dz/self.Rfg
		self.F3 = self.dt/self.Vg/BheData['capG']
		self.F4 = self.dz/self.Rgg1
		self.F5 = self.dt/self.dz*self.u
		self.F6 = self.dt/self.Vp/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag		
		self.F9 = 4./self.Rgs
		
		self.F6F2 = self.F6*self.F2
		self.F3F1 = self.F3*self.F1
		self.F3F2 = self.F3*self.F2
		self.F3F4 = self.F3*self.F4
		self.F3F8 = self.F3*self.F8
	
		# Variables NoFlow
		self.F5NoFlow = 0
		self.F2NoFlow = self.dz/self.RfgNoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow
		self.F6F2NoFlow = self.F6*self.F2NoFlow		
		
		# flow dependent variabls
		self.flow_vari = np.zeros(6)
		self.flow_vari[0] = (1 + self.F5 + self.F6F2)
		self.flow_vari[1] = (- self.F5)
		self.flow_vari[2] = (- self.F6F2)
		self.flow_vari[3] = (1 + 2*self.F3F8 + self.F3F1 + self.F3F2 + 2*self.F3F4)
		self.flow_vari[4] = (1 +   self.F3F8 + self.F3F1 + self.F3F2 + 2*self.F3F4)
		self.flow_vari[5] = (- self.F3F2)	
		
		# Set up Matrix
		self.na = 5 # number of cell arrays, Tin, Tout, Tgi, Tgo, Tsoil

		############### Matrix for flow #################
		#################################################
		##### Tgi
		## main diagonal Tgi 
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+2*self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## Tgi z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## Tgi z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## Tgi to soil 	
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## Tgi to tin
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)
		## Tgi to Tgo
		self.dataTemp = np.ones(self.nz)*-2*self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-4)*self.nz)

		##### Tgo
		## main diagonal Tgo 
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+2*self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # unterste Zelle
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)
		## Tgo z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+self.nz)
		## Tgo z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+self.nz)
		## Tgo to soil
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+4*self.nz)
		## Tgo to tout
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+3*self.nz)
		## Tgo to Tgi
		self.dataTemp = np.ones(self.nz)*-2*self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfi
		## main diagonal Tfi 
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.dataTemp[0] = 1
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tfi z-1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ 2*self.nz)
		## Tfi to Tgi 
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfo
		## main diagonal Tfo 
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)
		## Tfo z+1 
		self.data = np.append(self.data,np.ones(self.nz)*-self.F5)
		self.posXTemp = np.arange(0,self.nz,1)+ 3*self.nz
		self.posYTemp = np.arange(0,self.nz,1)+1+3*self.nz
		self.posYTemp[self.nz-1] = self.posYTemp[self.nz-1]-(self.nz+1)
		self.posX = np.append(self.posX,self.posXTemp)
		self.posY = np.append(self.posY,self.posYTemp)
		## Tfo to Tgo 
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)

		# main diagonal Soil 1
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 4*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 4*self.nz)

		self.K = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_Flow = sparse.csr_matrix(self.K)
		self.K_arr = self.K_sparse_Flow.toarray()
		
		# get indices of flow dependent variabels in martix
		self.vari_indices = []
		for vari in self.flow_vari:
			self.indx = [self.K_arr == vari]		
			self.vari_indices.append(self.indx) 
		
		############### Matrix for no flow #################
		####################################################
		self.F5 = 0
		#self.F2 = self.dz/self.RfgNoFlow
		self.F3F2 = self.F3*self.F2
		self.F6F2 = self.F6*self.F2
		##### Tgi
		## main diagonal Tgi 
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+2*self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## Tgi z-1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## Tgi z+1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## Tgi to soil		
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## Tgi to tin 
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)
		## Tgi to Tgo 
		self.dataTemp = np.ones(self.nz)*-2*self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-4)*self.nz)

		##### Tgo
		## main diagonal Tgo
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+2*self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+2*self.F3F4) # unterste Zelle
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)
		## Tgo z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+self.nz)
		## Tgo z+1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+self.nz)
		## Tgo to soil 
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+4*self.nz)
		## Tgo to tout 
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+3*self.nz)
		## Tgo to Tgi
		self.dataTemp = np.ones(self.nz)*-2*self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfi
		## main diagonal Tfi 
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.dataTemp[0] = 1
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tfi z-1 
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ 2*self.nz)
		## Tfi to Tgi
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfo
		## main diagonal Tfo
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)
		## Tfo z+1 
		self.data = np.append(self.data,np.ones(self.nz)*-self.F5)
		self.posXTemp = np.arange(0,self.nz,1)+ 3*self.nz
		self.posYTemp = np.arange(0,self.nz,1)+1+3*self.nz
		self.posYTemp[self.nz-1] = self.posYTemp[self.nz-1]-(self.nz+1)
		self.posX = np.append(self.posX,self.posXTemp)
		self.posY = np.append(self.posY,self.posYTemp)
		## Tfo to Tgo 
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)

		# main diagonal Soil 1
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 4*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 4*self.nz)

		self.K_NoFlow = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_NoFlow = sparse.csr_matrix(self.K_NoFlow)
		
			
		######## Talt ########
		######################
		self.result = np.ones(self.nz*self.na)*BheData['Tundist'] # Initialbedingung
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.T_grout_in = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		self.T_grout_out = np.full(self.nz,BheData['Tundist'])  # CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in	# Vorgabe Randbedingung Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
	
	
	def calcSondeFlowQ(self,tfinal,T_in,Q):	
		if Q != self.Qold:		
		#if Q > self.Qold*1.05 or Q < self.Qold*0.95:		
		#if True:
			self.Qold = Q	

			# Flow velocity
			self.u = self.Qold/(2*np.pi*self.rpi**2)
			
			# Flow parameters
			self.Pr = self.dynviscF*self.capF/self.densF/self.lmF
			self.Re = self.u*self.dpi/(self.dynviscF/self.densF)
			self.Nu = Nusselt2U (self.Re,self.Pr,self.rpi,self.length)
			self.Radv  = 1/(self.Nu*self.lmF*np.pi)			
			self.Rfg = self.Radv + self.Rcona + self.Rconb
								
			# Variables Flow					
			self.F2 = self.dz/self.Rfg
			self.F5 = self.dt/self.dz*self.u			
			self.F6F2 = self.F6*self.F2
			self.F3F2 = self.F3*self.F2
			
			self.flow_vari[0] = (1 + self.F5 + self.F6F2)
			self.flow_vari[1] = (- self.F5)
			self.flow_vari[2] = (- self.F6F2)
			self.flow_vari[3] = (1 + 2*self.F3F8 + self.F3F1 + self.F3F2 + 2*self.F3F4)
			self.flow_vari[4] = (1 +   self.F3F8 + self.F3F1 + self.F3F2 + 2*self.F3F4)
			self.flow_vari[5] = (- self.F3F2)	
			
			# Write updated Variables to matrix			
			for i in range(0,6):
				self.vari = self.vari_indices[i]				
				self.K_arr[tuple(self.vari)] = self.flow_vari[i]			
			self.K_sparse_Flow = sparse.csr_matrix(self.K_arr)
			
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in	# Vorgabe Randbedingung Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
	
	
	def calcSondeNoFlow(self,tfinal):	
		for i in range(0,tfinal):	
			self.U = sparse.linalg.spsolve(self.K_sparse_NoFlow,self.result)
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 			
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
		
class BHE_2U_expl:

	'''
	explicit simulation model for 2U bhe
	
	'''
		
	def setTimestep(self,dt):
		self.dt = dt	
		
	def setnz(self,nz):
		self.nz = nz
	
	def setSoilBC(self,Tsoil):
		self.TsA[:] = Tsoil
	
	def getGroutBC(self):
		return (np.mean(self.T_grout_in)+np.mean(self.T_grout_out))/2
	
	def getFluidOut(self):
		return self.Tf_out[0]
		
	def getFluidIn(self):
		return self.Tf_in[0]		
		
	def initialize(self,BheData):		
		self.dz = BheData['length']/self.nz
		self.dynviscF = BheData['dynviscF']
		
		# Geometry
		self.D = BheData['diamB']
		self.ro = BheData['diamB']/2
		self.rpi = BheData['odiamP']/2-BheData['thickP']
		self.rpo = BheData['odiamP']/2
		self.dpi = BheData['odiamP']-2*BheData['thickP']
		self.dpo = BheData['odiamP']
		
		# fluid properties
		self.dynviscF = BheData['dynviscF']
		self.densF = BheData['densF']
		self.lmF = BheData['lmF']
		self.length	= BheData['length']
		self.capF = BheData['capF']
		
		# Flow velocity
		self.u = BheData['Qf']/(2*np.pi*self.rpi**2)
		self.Qold = BheData['Qf']
		self.maxVel = self.u 
	
		# Flow parameters
		self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Re = self.u*self.dpi/(self.dynviscF/BheData['densF'])
		self.Nu = Nusselt2U (self.Re,self.Pr,self.rpi,BheData['length'])
		self.Radv  = 1/(self.Nu*BheData['lmF']*np.pi)
		
		
		# Resistances
		self.s = BheData['distP']*2**0.5
		self.x = np.log((self.D**2+4*self.dpo**2)**0.5/(2*2**0.5*self.dpo))/np.log(self.D/(2*self.dpo))
		self.Rg = (3.098-4.432*self.s/self.D + 2.364*self.s**2/(self.D**2)) * np.arccosh((self.D**2+self.dpo**2-self.s**2)/(2*self.D*self.dpo))/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg
		self.Rconb = self.x*self.Rg
		self.Rcona = np.log(self.rpo/self.rpi)/(2*np.pi*BheData['lmP'])
		self.Rar1 = np.arccosh((self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])
		self.Rar2 = np.arccosh((2*self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])
		self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
		
		### Check negative thermal resistances ###
		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):	
			self.x = 2/3*self.x	
			self.Rconb = self.x*self.Rg
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
			
			
		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):
			self.x = self.x*(1/3) 
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)

			
		if (((1/self.Rgg1 + 1/(2*self.Rgs))**(-1) <= 0) or ((1/self.Rgg2 + 1/(2*self.Rgs))**(-1) <= 0)):
			self.x = 0
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg1 = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			self.Rgg2 = (2*self.Rgs*(self.Rar2-2*self.x*self.Rg))/(2*self.Rgs-self.Rar2+2*self.x*self.Rg)
		
		
		self.Rfg = self.Radv + self.Rcona + self.Rconb		
		self.RfgNoFlow = self.Rconb + self.Rcona
		self.Rgs_coupling = 4./self.Rgs
		
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-4*self.rpo**2)/4	# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ap = np.pi*self.rpi**2					# area pipe/fluid
		self.Vp = self.Ap*self.dz						# volume pipe/fluid
			
		
		# Variables Flow					# Ab hier werden werte zusammengefasst damit rechnung schneller läuft
		self.F1 = self.dz/self.Rgs			# am besten zurücksubstituieren für verständnis
		self.F2 = self.dz/self.Rfg
		self.F3 = self.dt/self.Vg/BheData['capG']
		self.F4 = self.dz/self.Rgg1
		self.F5 = self.dt/self.dz*self.u
		self.F6 = self.dt/self.Vp/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag		
		
		self.F6F2 = self.F6*self.F2
		self.F3F1 = self.F3*self.F1
		self.F3F2 = self.F3*self.F2
		self.F3F4 = self.F3*self.F4
		self.F3F8 = self.F3*self.F8
	
		# Variables NoFlow
		self.F2NoFlow = self.dz/self.RfgNoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow
		self.F6F2NoFlow = self.F6*self.F2NoFlow
		
		self.idxPlus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),1)
		self.idxMinus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),-1)

		self.Q_cond = np.zeros(self.nz)
		self.Phi_w = np.zeros(self.nz)
		self.Phi_e = np.zeros(self.nz)
		
		# Calc Arrays
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 			# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 			# CellArray Tf_out
		self.T_grout_in = np.full(self.nz,BheData['Tundist'])  		# CellArray Tf_grout around pipe in
		self.T_grout_out = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout around pipe out
		self.TsA = np.full(self.nz,BheData['Tundist'])  			# CellArray Average Soil Temperature
		self.zgrid = np.linspace(0,BheData['length'],self.nz)	
		self.alt_Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.alt_Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.alt_T_grout_in = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		self.alt_T_grout_out = np.full(self.nz,BheData['Tundist'])  # CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout in
			self.Q_cond = np.take(self.alt_T_grout_in,self.idxMinus) - 2*self.alt_T_grout_in + np.take(self.alt_T_grout_in,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_in[1] - self.alt_T_grout_in[0]
			self.Q_cond[-1] = self.alt_T_grout_in[-2] - self.alt_T_grout_in[-1]
			self.T_grout_in = self.alt_T_grout_in + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_in) + self.F3F2 * (self.alt_Tf_in - self.alt_T_grout_in) + 2 * self.F3F4 * (self.alt_T_grout_out - self.alt_T_grout_in)
			
			# Berechnung grout out
			self.Q_cond = np.take(self.alt_T_grout_out,self.idxMinus) - 2*self.alt_T_grout_out + np.take(self.alt_T_grout_out,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_out[1] - self.alt_T_grout_out[0]
			self.Q_cond[-1] = self.alt_T_grout_out[-2] - self.alt_T_grout_out[-1]
			self.T_grout_out = self.alt_T_grout_out + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_out) + self.F3F2 * (self.alt_Tf_out - self.alt_T_grout_out) + 2 * self.F3F4 * (self.alt_T_grout_in - self.alt_T_grout_out)
			
			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Phi_w[0] = T_in
			self.Tf_in = self.alt_Tf_in + self.F5 * (self.Phi_w - self.alt_Tf_in) + self.F6F2 * (self.alt_T_grout_in - self.alt_Tf_in)
			
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_out,self.idxMinus)
			self.Phi_w[-1] = self.alt_Tf_in[-1]
			self.Tf_out = self.alt_Tf_out + self.F5 * (self.Phi_w - self.alt_Tf_out) + self.F6F2 * (self.alt_T_grout_out - self.alt_Tf_out)

			# Rückschreiben
			self.alt_T_grout_in[:] = self.T_grout_in[:]
			self.alt_T_grout_out[:] = self.T_grout_out[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]
		return True
		
	def calcSondeFlowQ(self,tfinal,T_in,Q):			
		#if Q != self.Qold:		
		if Q > self.Qold*1.05 or Q < self.Qold*0.95:		
			self.Qold = Q	

			# Flow velocity
			self.u = self.Qold/(2*np.pi*self.rpi**2)
			
			# Flow parameters
			self.Pr = self.dynviscF*self.capF/self.densF/self.lmF
			self.Re = self.u*self.dpi/(self.dynviscF/self.densF)
			self.Nu = Nusselt2U (self.Re,self.Pr,self.rpi,self.length)
			self.Radv  = 1/(self.Nu*self.lmF*np.pi)			
			self.Rfg = self.Radv + self.Rcona + self.Rconb
			
			# Variables Flow					
			self.F2 = self.dz/self.Rfg
			self.F5 = self.dt/self.dz*self.u			
			self.F6F2 = self.F6*self.F2
			self.F3F2 = self.F3*self.F2
		
		
		for i in range(0,tfinal):
			# Berechnung grout in
			self.Q_cond = np.take(self.alt_T_grout_in,self.idxMinus) - 2*self.alt_T_grout_in + np.take(self.alt_T_grout_in,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_in[1] - self.alt_T_grout_in[0]
			self.Q_cond[-1] = self.alt_T_grout_in[-2] - self.alt_T_grout_in[-1]
			self.T_grout_in = self.alt_T_grout_in + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_in) + self.F3F2 * (self.alt_Tf_in - self.alt_T_grout_in) + 2 * self.F3F4 * (self.alt_T_grout_out - self.alt_T_grout_in)
			
			# Berechnung grout out
			self.Q_cond = np.take(self.alt_T_grout_out,self.idxMinus) - 2*self.alt_T_grout_out + np.take(self.alt_T_grout_out,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_out[1] - self.alt_T_grout_out[0]
			self.Q_cond[-1] = self.alt_T_grout_out[-2] - self.alt_T_grout_out[-1]
			self.T_grout_out = self.alt_T_grout_out + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_out) + self.F3F2 * (self.alt_Tf_out - self.alt_T_grout_out) + 2 * self.F3F4 * (self.alt_T_grout_in - self.alt_T_grout_out)
			
			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Phi_w[0] = T_in
			self.Tf_in = self.alt_Tf_in + self.F5 * (self.Phi_w - self.alt_Tf_in) + self.F6F2 * (self.alt_T_grout_in - self.alt_Tf_in)
			
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_out,self.idxMinus)
			self.Phi_w[-1] = self.alt_Tf_in[-1]
			self.Tf_out = self.alt_Tf_out + self.F5 * (self.Phi_w - self.alt_Tf_out) + self.F6F2 * (self.alt_T_grout_out - self.alt_Tf_out)

			# Rückschreiben
			self.alt_T_grout_in[:] = self.T_grout_in[:]
			self.alt_T_grout_out[:] = self.T_grout_out[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]
		return True	
		
	def calcSondeNoFlow(self,tfinal):	
		for i in range(0,tfinal):
			# Berechnung grout in
			self.Q_cond = np.take(self.alt_T_grout_in,self.idxMinus) - 2*self.alt_T_grout_in + np.take(self.alt_T_grout_in,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_in[1] - self.alt_T_grout_in[0]
			self.Q_cond[-1] = self.alt_T_grout_in[-2] - self.alt_T_grout_in[-1]
			self.T_grout_in = self.alt_T_grout_in + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_in) + self.F3F2NoFlow * (self.alt_Tf_in - self.alt_T_grout_in) + 2*self.F3F4 * (self.alt_T_grout_out - self.alt_T_grout_in)
			
			# Berechnung grout out
			self.Q_cond = np.take(self.alt_T_grout_out,self.idxMinus) - 2*self.alt_T_grout_out + np.take(self.alt_T_grout_out,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_out[1] - self.alt_T_grout_out[0]
			self.Q_cond[-1] = self.alt_T_grout_out[-2] - self.alt_T_grout_out[-1]
			self.T_grout_out = self.alt_T_grout_out + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_out) + self.F3F2NoFlow * (self.alt_Tf_out - self.alt_T_grout_out) + 2*self.F3F4 * (self.alt_T_grout_in - self.alt_T_grout_out)
			
			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Tf_in = self.alt_Tf_in + self.F6F2NoFlow *2* (self.alt_T_grout_in - self.alt_Tf_in)
			
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_in,self.idxMinus)
			self.Phi_w[-1] = self.alt_Tf_in[-1]
			self.Tf_out = self.alt_Tf_out + self.F6F2NoFlow *2* (self.alt_T_grout_out - self.alt_Tf_out)
		
			# Rückschreiben
			self.alt_T_grout_in[:] = self.T_grout_in[:]
			self.alt_T_grout_out[:] = self.T_grout_out[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]	
		return True

class BHE_1U_expl:

	'''
	explicit simulation model for 1U bhe
	
	'''
		
	def setTimestep(self,dt):
		self.dt = dt	
		
	def setnz(self,nz):
		self.nz = nz
	
	def setSoilBC(self,Tsoil):
		self.TsA[:] = Tsoil
	
	def getGroutBC(self):
		return (np.mean(self.T_grout_in)+np.mean(self.T_grout_out))/2
	
	def getFluidOut(self):
		return self.Tf_out[0]
		
	def getFluidIn(self):
		return self.Tf_in[0]		
		
	def initialize(self,BheData):		
		self.dz = BheData['length']/self.nz
		self.dynviscF = BheData['dynviscF']
		
		# Geometry
		self.D = BheData['diamB']
		self.ro = BheData['diamB']/2
		self.rpi = BheData['odiamP']/2-BheData['thickP']
		self.rpo = BheData['odiamP']/2
		self.dpi = BheData['odiamP']-2*BheData['thickP']
		self.dpo = BheData['odiamP']
		
		# Flow velocity
		self.u = BheData['Qf']/(np.pi*self.rpi**2)
		self.maxVel = self.u 
	
		# Flow parameters
		self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Re = self.u*self.dpi/(self.dynviscF/BheData['densF'])
		self.Nu = Nusselt1U (self.Re,self.Pr,self.rpi,BheData['length'])
		self.Radv  = 1/(self.Nu*BheData['lmF']*np.pi)
		
		
		# Resistances
		self.s = BheData['distP']
		self.x = np.log((self.D**2+2*self.dpo**2)**0.5/(2*self.dpo))/np.log(self.D/(2**0.5*self.dpo))
		self.Rg = (1.601-0.888*self.s/self.D) * np.arccosh((self.D**2+self.dpo**2-self.s**2)/(2*self.D*self.dpo))/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg		
		self.Rconb = self.x*self.Rg		
		self.Rcona = np.log(self.rpo/self.rpi)/(2*np.pi*BheData['lmP'])	
		self.Rar1 = np.arccosh((2*self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])		
		self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)

		
		### Check negative thermal resistances ###
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = 2/3*self.x	
			self.Rconb = self.x*self.Rg
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = self.x * 1/3	
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = 0	
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)	
		self.Rfg = self.Radv + self.Rcona + self.Rconb
		self.RfgNoFlow = self.Rconb + self.Rcona
		self.Rgs_coupling = 2./self.Rgs
		
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-2*self.rpo**2)/2	# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ap = np.pi*self.rpi**2						# area pipe/fluid
		self.Vp = self.Ap*self.dz						# volume pipe/fluid
			
		
		# Variables Flow					# Ab hier werden werte zusammengefasst damit rechnung schneller läuft
		self.F1 = self.dz/self.Rgs			# am besten zurücksubstituieren für verständnis
		self.F2 = self.dz/self.Rfg
		self.F3 = self.dt/self.Vg/BheData['capG']
		self.F4 = self.dz/self.Rgg
		self.F5 = self.dt/self.dz*self.u
		self.F6 = self.dt/self.Vp/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag		
		
		self.F6F2 = self.F6*self.F2
		self.F3F1 = self.F3*self.F1
		self.F3F2 = self.F3*self.F2
		self.F3F4 = self.F3*self.F4
		self.F3F8 = self.F3*self.F8
	
		# Variables NoFlow
		self.F2NoFlow = self.dz/self.RfgNoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow
		self.F6F2NoFlow = self.F6*self.F2NoFlow
		
		self.idxPlus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),1)
		self.idxMinus = np.roll(np.linspace(0,self.nz-1,self.nz,dtype = 'int'),-1)

		self.Q_cond = np.zeros(self.nz)
		self.Phi_w = np.zeros(self.nz)
		self.Phi_e = np.zeros(self.nz)
		
		# Calc Arrays
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 			# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 			# CellArray Tf_out
		self.T_grout_in = np.full(self.nz,BheData['Tundist'])  		# CellArray Tf_grout around pipe in
		self.T_grout_out = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout around pipe out
		self.TsA = np.full(self.nz,BheData['Tundist'])  			# CellArray Average Soil Temperature
		self.zgrid = np.linspace(0,BheData['length'],self.nz)	
		self.alt_Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.alt_Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.alt_T_grout_in = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		self.alt_T_grout_out = np.full(self.nz,BheData['Tundist'])  # CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout in
			self.Q_cond = np.take(self.alt_T_grout_in,self.idxMinus) - 2*self.alt_T_grout_in + np.take(self.alt_T_grout_in,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_in[1] - self.alt_T_grout_in[0]
			self.Q_cond[-1] = self.alt_T_grout_in[-2] - self.alt_T_grout_in[-1]
			self.T_grout_in = self.alt_T_grout_in + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_in) + self.F3F2 * (self.alt_Tf_in - self.alt_T_grout_in) +   self.F3F4 * (self.alt_T_grout_out - self.alt_T_grout_in)
			
			# Berechnung grout out
			self.Q_cond = np.take(self.alt_T_grout_out,self.idxMinus) - 2*self.alt_T_grout_out + np.take(self.alt_T_grout_out,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_out[1] - self.alt_T_grout_out[0]
			self.Q_cond[-1] = self.alt_T_grout_out[-2] - self.alt_T_grout_out[-1]
			self.T_grout_out = self.alt_T_grout_out + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_out) + self.F3F2 * (self.alt_Tf_out - self.alt_T_grout_out) +  self.F3F4 * (self.alt_T_grout_in - self.alt_T_grout_out)
			
			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Phi_w[0] = T_in
			self.Tf_in = self.alt_Tf_in + self.F5 * (self.Phi_w - self.alt_Tf_in) + self.F6F2 * (self.alt_T_grout_in - self.alt_Tf_in)
			
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_out,self.idxMinus)
			self.Phi_w[-1] = self.alt_Tf_in[-1]
			self.Tf_out = self.alt_Tf_out + self.F5 * (self.Phi_w - self.alt_Tf_out) + self.F6F2 * (self.alt_T_grout_out - self.alt_Tf_out)

			# Rückschreiben
			self.alt_T_grout_in[:] = self.T_grout_in[:]
			self.alt_T_grout_out[:] = self.T_grout_out[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]
		return True
		
	def calcSondeNoFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			# Berechnung grout in
			self.Q_cond = np.take(self.alt_T_grout_in,self.idxMinus) - 2*self.alt_T_grout_in + np.take(self.alt_T_grout_in,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_in[1] - self.alt_T_grout_in[0]
			self.Q_cond[-1] = self.alt_T_grout_in[-2] - self.alt_T_grout_in[-1]
			self.T_grout_in = self.alt_T_grout_in + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_in) + self.F3F2NoFlow * (self.alt_Tf_in - self.alt_T_grout_in) + self.F3F4 * (self.alt_T_grout_out - self.alt_T_grout_in)
			
			# Berechnung grout out
			self.Q_cond = np.take(self.alt_T_grout_out,self.idxMinus) - 2*self.alt_T_grout_out + np.take(self.alt_T_grout_out,self.idxPlus)
			self.Q_cond[0] = self.alt_T_grout_out[1] - self.alt_T_grout_out[0]
			self.Q_cond[-1] = self.alt_T_grout_out[-2] - self.alt_T_grout_out[-1]
			self.T_grout_out = self.alt_T_grout_out + self.F3F8 * self.Q_cond + self.F3F1 * (self.TsA - self.alt_T_grout_out) + self.F3F2NoFlow * (self.alt_Tf_out - self.alt_T_grout_out) + self.F3F4 * (self.alt_T_grout_in - self.alt_T_grout_out)
			
			# Berechnung fluid in
			self.Phi_w = np.take(self.alt_Tf_in,self.idxPlus)
			self.Tf_in = self.alt_Tf_in + self.F6F2NoFlow *2* (self.alt_T_grout_in - self.alt_Tf_in)
			
			# Berechnung fluid out	
			self.Phi_w = np.take(self.alt_Tf_in,self.idxMinus)
			self.Phi_w[-1] = self.alt_Tf_in[-1]
			self.Tf_out = self.alt_Tf_out + self.F6F2NoFlow *2* (self.alt_T_grout_out - self.alt_Tf_out)
		
			# Rückschreiben
			self.alt_T_grout_in[:] = self.T_grout_in[:]
			self.alt_T_grout_out[:] = self.T_grout_out[:]
			self.alt_Tf_in[:] = self.Tf_in[:]
			self.alt_Tf_out[:] = self.Tf_out[:]	
		return True

class BHE_1U_impl:
	
	'''
	implicit simulation model for 1U bhe
	
	'''
	
	def setSoilBC(self,Tsoil):
		self.result[4*self.nz:5*self.nz] = Tsoil
	
	def getGroutBC(self):
		return (np.mean(self.T_grout_in[1:])+np.mean(self.T_grout_out[1:]))/2
	
	def getFluidOut(self):
		return self.Tf_out[1]
		
	def getFluidIn(self):
		return self.Tf_in[1]	

	def setTimestep(self,dt):
		self.dt = dt	
		
	def setnz(self,nz):
		self.nz = nz
		
	def initialize(self,BheData):		
		self.dz = BheData['length']/self.nz
		self.nz = self.nz+1
		
		# Geometry
		self.D = BheData['diamB']
		self.ro = BheData['diamB']/2
		self.rpi = BheData['odiamP']/2-BheData['thickP']
		self.rpo = BheData['odiamP']/2
		self.dpi = BheData['odiamP']-2*BheData['thickP']
		self.dpo = BheData['odiamP']
			
		# Flow velocity
		self.dynviscF = BheData['dynviscF']
		self.u = BheData['Qf']/(np.pi*self.rpi**2)
		self.Qold = BheData['Qf']
		self.maxVel = self.u 
	
		# Flow parameters
		self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
		self.Re = self.u*self.dpi/(self.dynviscF/BheData['densF'])
		self.Nu = Nusselt1U (self.Re,self.Pr,self.rpi,BheData['length'])
		self.Radv  = 1/(self.Nu*BheData['lmF']*np.pi)
		
		
		# Resistances
		self.s = BheData['distP']
		self.x = np.log((self.D**2+2*self.dpo**2)**0.5/(2*self.dpo))/np.log(self.D/(2**0.5*self.dpo))
		self.Rg = (1.601-0.888*self.s/self.D) * np.arccosh((self.D**2+self.dpo**2-self.s**2)/(2*self.D*self.dpo))/(2*np.pi*BheData['lmG'])
		self.Rgs = (1-self.x)*self.Rg		
		self.Rconb = self.x*self.Rg		
		self.Rcona = np.log(self.rpo/self.rpi)/(2*np.pi*BheData['lmP'])	
		self.Rar1 = np.arccosh((2*self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])		
		self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)

		
		### Check negative thermal resistances ###
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = 2/3*self.x	
			self.Rconb = self.x*self.Rg
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = self.x * 1/3	
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
		if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
			self.x = 0	
			self.Rconb = self.x*self.Rg	
			self.Rgs = (1-self.x)*self.Rg
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)	
		self.Rfg = self.Radv + self.Rcona + self.Rconb		
		self.RfgNoFlow = self.Rconb + self.Rcona
		self.Rgs_coupling = 2./self.Rgs
				
		# Volumes and Areas
		self.Ag = np.pi*(self.ro**2-2*self.rpo**2)/2	# area grout
		self.Vg = self.Ag*self.dz						# volume grout
		self.Ap = np.pi*self.rpi**2						# area pipe/fluid
		self.Vp = self.Ap*self.dz						# volume pipe/fluid
		
		# Variables Flow					# Ab hier werden werte zusammengefasst damit rechnung schneller läuft
		self.F1 = self.dz/self.Rgs			# am besten zurücksubstituieren für verständnis
		self.F2 = self.dz/self.Rfg
		self.F3 = self.dt/self.Vg/BheData['capG']
		self.F4 = self.dz/self.Rgg
		self.F5 = self.dt/self.dz*self.u
		self.F6 = self.dt/self.Vp/BheData['capF']
		self.F8 = BheData['lmG']/self.dz*self.Ag		

		
		self.F6F2 = self.F6*self.F2
		self.F3F1 = self.F3*self.F1
		self.F3F2 = self.F3*self.F2
		self.F3F4 = self.F3*self.F4
		self.F3F8 = self.F3*self.F8
	
		# Variables NoFlow
		self.F5NoFlow = 0
		self.F2NoFlow = self.dz/self.RfgNoFlow
		self.F3F2NoFlow = self.F3*self.F2NoFlow
		self.F6F2NoFlow = self.F6*self.F2NoFlow	

		# flow dependent variabls
		self.flow_vari = np.zeros(6)
		self.flow_vari[0] = (1 + self.F5 + self.F6F2)
		self.flow_vari[1] = (- self.F5)
		self.flow_vari[2] = (- self.F6F2)
		self.flow_vari[3] = (1 + 2*self.F3F8 + self.F3F1 + self.F3F2 +   self.F3F4)
		self.flow_vari[4] = (1 +   self.F3F8 + self.F3F1 + self.F3F2 +   self.F3F4)
		self.flow_vari[5] = (- self.F3F2)		
		
		# Set up Matrix
		self.na = 5 # number of cell arrays, Tin, Tout, Tgi, Tgo, Tsoil

		
		############### Matrix for flow #################
		#################################################
		##### Tgi
		## main diagonal Tgi 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## Tgi z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## Tgi z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## Tgi to soil -F3F1 = -2		
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## Tgi to tin -F3F12 = -3
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)
		## Tgi to Tgo -2F3F14 = -8
		self.dataTemp = np.ones(self.nz)*-self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-4)*self.nz)

		##### Tgo
		## main diagonal Tgo 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)
		## Tgo z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+self.nz)
		## Tgo z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+self.nz)
		## Tgo to soil -F3F1 = -2
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+4*self.nz)
		## Tgo to tout -F3F12 = -3
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+3*self.nz)
		## Tgo to Tgi -2F3F14 = -8
		self.dataTemp = np.ones(self.nz)*-self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfi
		## main diagonal Tfi 1+F5+F6F2 = 12, oberste Zelle 1+F6F2 = 7
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.dataTemp[0] = 1
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tfi z-1 -F5 = -5, oberste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ 2*self.nz)
		## Tfi to Tgi -F6F2 = -6
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfo
		## main diagonal Tfo 1+F5+F6F2 = 12, 
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)
		## Tfo z+1 -F5 = -5, unterste Zelle Tfi statt Tfo
		self.data = np.append(self.data,np.ones(self.nz)*-self.F5)
		self.posXTemp = np.arange(0,self.nz,1)+ 3*self.nz
		self.posYTemp = np.arange(0,self.nz,1)+1+3*self.nz
		self.posYTemp[self.nz-1] = self.posYTemp[self.nz-1]-(self.nz+1)
		self.posX = np.append(self.posX,self.posXTemp)
		self.posY = np.append(self.posY,self.posYTemp)
		## Tfo to Tgo -F6F2 = -6
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)

		# main diagonal Soil 1
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 4*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 4*self.nz)

		self.K = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_Flow = sparse.csr_matrix(self.K)
		self.K_arr = self.K_sparse_Flow.toarray()
		
		# get indices of flow dependent variabels in martix
		self.vari_indices = []
		for vari in self.flow_vari:
			self.indx = [self.K_arr == vari]		
			self.vari_indices.append(self.indx) 
		
		############### Matrix for no flow #################
		####################################################
		self.F5 = 0
		#self.F2 = self.dz/self.RfgNoFlow
		self.F3F2 = self.F3*self.F2
		self.F6F2 = self.F6*self.F2
		##### Tgi
		## main diagonal Tgi 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
		self.data = self.dataTemp
		self.posX = np.arange(0,self.nz,1)
		self.posY = np.arange(0,self.nz,1)
		## Tgi z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
		## Tgi z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
		## Tgi to soil -F3F1 = -2		
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
		## Tgi to tin -F3F12 = -3
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)
		## Tgi to Tgo -2F3F14 = -8
		self.dataTemp = np.ones(self.nz)*-self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1))
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-4)*self.nz)

		##### Tgo
		## main diagonal Tgo 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
		self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
		self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
		self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)
		## Tgo z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+self.nz)
		## Tgo z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+self.nz)
		## Tgo to soil -F3F1 = -2
		self.dataTemp = np.ones(self.nz)*-self.F3F1
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+4*self.nz)
		## Tgo to tout -F3F12 = -3
		self.dataTemp = np.ones(self.nz)*-self.F3F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+3*self.nz)
		## Tgo to Tgi -2F3F14 = -8
		self.dataTemp = np.ones(self.nz)*-self.F3F4
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfi
		## main diagonal Tfi 1+F5+F6F2 = 12, oberste Zelle 1+F6F2 = 7
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.dataTemp[0] = 1
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
		## Tfi z-1 -F5 = -5, oberste Zelle = 0
		self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5)
		self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ 2*self.nz)
		## Tfi to Tgi -F6F2 = -6
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 2*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1))

		##### Tfo
		## main diagonal Tfo 1+F5+F6F2 = 12, 
		self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)
		## Tfo z+1 -F5 = -5, unterste Zelle Tfi statt Tfo
		self.data = np.append(self.data,np.ones(self.nz)*-self.F5)
		self.posXTemp = np.arange(0,self.nz,1)+ 3*self.nz
		self.posYTemp = np.arange(0,self.nz,1)+1+3*self.nz
		self.posYTemp[self.nz-1] = self.posYTemp[self.nz-1]-(self.nz+1)
		self.posX = np.append(self.posX,self.posXTemp)
		self.posY = np.append(self.posY,self.posYTemp)
		## Tfo to Tgo -F6F2 = -6
		self.dataTemp = np.ones(self.nz)*-self.F6F2
		self.dataTemp[0] = 0
		self.data = np.append(self.data,self.dataTemp)
		self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 3*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)

		# main diagonal Soil 1
		self.data = np.append(self.data,np.ones(self.nz))
		self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 4*self.nz)
		self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 4*self.nz)

		self.K_NoFlow = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
		self.K_sparse_NoFlow = sparse.csr_matrix(self.K_NoFlow)
		
			
		######## Talt ########
		######################
		self.result = np.ones(self.nz*self.na)*BheData['Tundist'] # Initialbedingung
		self.Tf_in = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_in
		self.Tf_out = np.full(self.nz,BheData['Tundist']) 		# CellArray Tf_out
		self.T_grout_in = np.full(self.nz,BheData['Tundist'])  	# CellArray Tf_grout
		self.T_grout_out = np.full(self.nz,BheData['Tundist'])  # CellArray Tf_grout
		return True
	
	def calcSondeFlow(self,tfinal,T_in):	
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in	# Vorgabe Randbedingung Tin
			#self.U = sparse.linalg.bicgstab(self.K_sparse_Flow, self.result, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)[0]
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
		
	def calcSondeFlowAllOut(self,tfinal,T_in):	
		self.Tf_AllOut = np.zeros(tfinal)
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in[i]	# Vorgabe Randbedingung Tin
			#self.U = sparse.linalg.bicgstab(self.K_sparse_Flow, self.result, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)[0]
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
			self.Tf_AllOut[i] = self.Tf_out[1]
		return True
	
	def calcSondeFlowQ(self,tfinal,T_in,Q):	
		if Q != self.Qold:		
		#if Q > self.Qold*1.05 or Q < self.Qold*0.95:		
		#if True:
			self.Qold = Q	


			# Flow velocity
			self.u = self.Qold/(np.pi*self.rpi**2)
			
			# Flow parameters
			self.Pr = self.dynviscF*self.capF/self.densF/self.lmF
			self.Re = self.u*self.dpi/(self.dynviscF/self.densF)
			self.Nu = Nusselt1U (self.Re,self.Pr,self.rpi,self.length)
			self.Radv  = 1/(self.Nu*self.lmF*np.pi)			
			self.Rfg = self.Radv + self.Rcona + self.Rconb
								
			# Variables Flow					
			self.F2 = self.dz/self.Rfg
			self.F5 = self.dt/self.dz*self.u			
			self.F6F2 = self.F6*self.F2
			self.F3F2 = self.F3*self.F2
			
			self.flow_vari[0] = (1 + self.F5 + self.F6F2)
			self.flow_vari[1] = (- self.F5)
			self.flow_vari[2] = (- self.F6F2)
			self.flow_vari[3] = (1 + 2*self.F3F8 + self.F3F1 + self.F3F2 + self.F3F4)
			self.flow_vari[4] = (1 +   self.F3F8 + self.F3F1 + self.F3F2 + self.F3F4)
			self.flow_vari[5] = (- self.F3F2)	
			
			# Write updated Variables to matrix			
			for i in range(0,6):
				self.vari = self.vari_indices[i]				
				self.K_arr[tuple(self.vari)] = self.flow_vari[i]			
			self.K_sparse_Flow = sparse.csr_matrix(self.K_arr)
			
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in	# Vorgabe Randbedingung Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
	'''	
	def calcSondeFlowQ(self,tfinal,T_in,Q,BheData):	
		if Q != self.Qold:
			
			self.Qold = Q	

			# Flow velocity
			self.dynviscF = BheData['dynviscF']
			self.u = BheData['Qf']/(np.pi*self.rpi**2)
			self.maxVel = self.u 

			# Flow parameters
			self.Pr = self.dynviscF*BheData['capF']/BheData['densF']/BheData['lmF']
			self.Re = self.u*self.dpi/(self.dynviscF/BheData['densF'])
			self.Nu = Nusselt1U (self.Re,self.Pr,self.rpi,BheData['length'])
			self.Radv  = 1/(self.Nu*BheData['lmF']*np.pi)
			
			
			# Resistances
			self.s = BheData['distP']
			self.x = np.log((self.D**2+2*self.dpo**2)**0.5/(2*self.dpo))/np.log(self.D/(2**0.5*self.dpo))
			self.Rg = (1.601-0.888*self.s/self.D) * np.arccosh((self.D**2+self.dpo**2-self.s**2)/(2*self.D*self.dpo))/(2*np.pi*BheData['lmG'])
			self.Rgs = (1-self.x)*self.Rg		
			self.Rconb = self.x*self.Rg		
			self.Rcona = np.log(self.rpo/self.rpi)/(2*np.pi*BheData['lmP'])	
			self.Rar1 = np.arccosh((2*self.s**2-self.dpo**2)/self.dpo**2)/(2*np.pi*BheData['lmG'])		
			self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)

			
			### Check negative thermal resistances ###
			if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
				self.x = 2/3*self.x	
				self.Rconb = self.x*self.Rg
				self.Rgs = (1-self.x)*self.Rg
				self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
				self.x = self.x * 1/3	
				self.Rconb = self.x*self.Rg	
				self.Rgs = (1-self.x)*self.Rg
				self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)
			if ((1/self.Rgg + 1/(2*self.Rgs))**(-1) <= 0):
				self.x = 0	
				self.Rconb = self.x*self.Rg	
				self.Rgs = (1-self.x)*self.Rg
				self.Rgg = (2*self.Rgs*(self.Rar1-2*self.x*self.Rg))/(2*self.Rgs-self.Rar1+2*self.x*self.Rg)	
			self.Rfg = self.Radv + self.Rcona + self.Rconb		
			self.RfgNoFlow = self.Rconb + self.Rcona
			self.Rgs_coupling = 2./self.Rgs
					
			# Volumes and Areas
			self.Ag = np.pi*(self.ro**2-2*self.rpo**2)/2	# area grout
			self.Vg = self.Ag*self.dz						# volume grout
			self.Ap = np.pi*self.rpi**2						# area pipe/fluid
			self.Vp = self.Ap*self.dz						# volume pipe/fluid
			
			# Variables Flow					# Ab hier werden werte zusammengefasst damit rechnung schneller läuft
			self.F1 = self.dz/self.Rgs			# am besten zurücksubstituieren für verständnis
			self.F2 = self.dz/self.Rfg
			self.F3 = self.dt/self.Vg/BheData['capG']
			self.F4 = self.dz/self.Rgg
			self.F5 = self.dt/self.dz*self.u
			self.F6 = self.dt/self.Vp/BheData['capF']
			self.F8 = BheData['lmG']/self.dz*self.Ag		
			
			self.F6F2 = self.F6*self.F2
			self.F3F1 = self.F3*self.F1
			self.F3F2 = self.F3*self.F2
			self.F3F4 = self.F3*self.F4
			self.F3F8 = self.F3*self.F8
		
			# Variables NoFlow
			self.F5NoFlow = 0
			self.F2NoFlow = self.dz/self.RfgNoFlow
			self.F3F2NoFlow = self.F3*self.F2NoFlow
			self.F6F2NoFlow = self.F6*self.F2NoFlow		
			
			# Set up Matrix
			self.na = 5 # number of cell arrays, Tin, Tout, Tgi, Tgo, Tsoil

			# Reihenfolge in Matrix: Tgi, Tgo, Tin, Tout, TSoil
			# data = Datenarray, posX = X Position, posY = Y Position, dataTemp = Temporär

			#self.result = np.ones(self.nz*self.na)*self.Tundist # Initialbedingung
			#self.result[2*nz] = Tin	# Vorlauftemperatur
			#self.result[0:nz] =  Tgi
			#self.result[nz:2*nz] = Tgo
			#self.result[2*nz:3*nz] = Tin
			#self.result[3*nz:4*nz] = Tout
			#self.result[4*nz:5*nz] = Tsoil
			
			############### Matrix for flow #################
			#################################################
			##### Tgi
			## main diagonal Tgi 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
			self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
			self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
			self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
			self.data = self.dataTemp
			self.posX = np.arange(0,self.nz,1)
			self.posY = np.arange(0,self.nz,1)
			## Tgi z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
			self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
			self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1)
			self.posY = np.append(self.posY,np.arange(0,self.nz-1,1))
			## Tgi z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
			self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
			self.posX = np.append(self.posX,np.arange(0,self.nz-1,1))
			self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1)
			## Tgi to soil -F3F1 = -2		
			self.dataTemp = np.ones(self.nz)*-self.F3F1
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1))
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-1)*self.nz)
			## Tgi to tin -F3F12 = -3
			self.dataTemp = np.ones(self.nz)*-self.F3F2
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1))
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-3)*self.nz)
			## Tgi to Tgo -2F3F14 = -8
			self.dataTemp = np.ones(self.nz)*-self.F3F4
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1))
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+(self.na-4)*self.nz)

			##### Tgo
			## main diagonal Tgo 1+2F3F8+F3F1+F3F2+2F3F4 = 20, oberste und unterste Zelle grout 1+1F3F8+F3F1+F3F2+2F3F4 = 19
			self.dataTemp = np.ones(self.nz)*(1+2*self.F3F8+self.F3F1+self.F3F2+self.F3F4)
			self.dataTemp[0] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # oberste Zelle
			self.dataTemp[self.nz-1] = (1+self.F3F8+self.F3F1+self.F3F2+self.F3F4) # unterste Zelle
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)
			## Tgo z-1 -F3F8 = -1, oberste Zelle = 0, unterste Zelle = -F3F8 = -1
			self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
			self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+self.nz)
			## Tgo z+1 -F3F8 = -1, oberste Zelle = -F3F8 = -1, unterste Zelle = 0
			self.data = np.append(self.data,np.ones(self.nz-1)*-self.F3F8)
			self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+1+self.nz)
			## Tgo to soil -F3F1 = -2
			self.dataTemp = np.ones(self.nz)*-self.F3F1
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+4*self.nz)
			## Tgo to tout -F3F12 = -3
			self.dataTemp = np.ones(self.nz)*-self.F3F2
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+3*self.nz)
			## Tgo to Tgi -2F3F14 = -8
			self.dataTemp = np.ones(self.nz)*-self.F3F4
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1))

			##### Tfi
			## main diagonal Tfi 1+F5+F6F2 = 12, oberste Zelle 1+F6F2 = 7
			self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
			self.dataTemp[0] = 1
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 2*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 2*self.nz)
			## Tfi z-1 -F5 = -5, oberste Zelle = 0
			self.data = np.append(self.data,np.ones(self.nz-1)*-self.F5)
			self.posX = np.append(self.posX,np.arange(0,self.nz-1,1)+1+ 2*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz-1,1)+ 2*self.nz)
			## Tfi to Tgi -F6F2 = -6
			self.dataTemp = np.ones(self.nz)*-self.F6F2
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 2*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1))

			##### Tfo
			## main diagonal Tfo 1+F5+F6F2 = 12, 
			self.dataTemp = np.ones(self.nz)*(1+self.F5+self.F6F2)
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 3*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 3*self.nz)
			## Tfo z+1 -F5 = -5, unterste Zelle Tfi statt Tfo
			self.data = np.append(self.data,np.ones(self.nz)*-self.F5)
			self.posXTemp = np.arange(0,self.nz,1)+ 3*self.nz
			self.posYTemp = np.arange(0,self.nz,1)+1+3*self.nz
			self.posYTemp[self.nz-1] = self.posYTemp[self.nz-1]-(self.nz+1)
			self.posX = np.append(self.posX,self.posXTemp)
			self.posY = np.append(self.posY,self.posYTemp)
			## Tfo to Tgo -F6F2 = -6
			self.dataTemp = np.ones(self.nz)*-self.F6F2
			self.dataTemp[0] = 0
			self.data = np.append(self.data,self.dataTemp)
			self.posX = np.append(self.posX,np.arange(0,self.nz,1)+ 3*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1)+self.nz)

			# main diagonal Soil 1
			self.data = np.append(self.data,np.ones(self.nz))
			self.posX = np.append(self.posX,np.arange(0,self.nz,1) + 4*self.nz)
			self.posY = np.append(self.posY,np.arange(0,self.nz,1) + 4*self.nz)

			self.K = csr_matrix((self.data, (self.posX, self.posY)), shape=(self.nz*self.na, self.nz*self.na),dtype=np.float)
			self.K_sparse_Flow = sparse.csr_matrix(self.K)

			
		for i in range(0,tfinal):
			self.result[2*self.nz] = T_in	# Vorgabe Randbedingung Tin
			self.U =sparse.linalg.spsolve(self.K_sparse_Flow,self.result)	# Lösen GlS
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
	'''
	def calcSondeNoFlow(self,tfinal):	
		for i in range(0,tfinal):	
			self.U = sparse.linalg.spsolve(self.K_sparse_NoFlow,self.result)
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 			
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
		return True
		
	def calcSondeNoFlowAllOut(self,tfinal,T_in):	
		self.Tf_AllOut = np.zeros(tfinal)
		for i in range(0,tfinal):	
			self.U = sparse.linalg.spsolve(self.K_sparse_NoFlow,self.result)
			# Rausschreiben für Plot
			self.T_grout_in[:] = self.U[0:self.nz] 
			self.T_grout_out[:] = self.U[self.nz:2*self.nz] 
			self.Tf_in = self.U[2*self.nz:3*self.nz] 
			self.Tf_out = self.U[3*self.nz:4*self.nz] 			
			# reinschreiben nächster Zeitschritt
			self.result[:] = self.U[:] 
			self.Tf_AllOut[i] = self.Tf_out[1]
		return True

			
def Nusselt2U (Re,Pr,rpi,H):
	'''
	calculate Nusselt Number for 2U BHE
	inputs: Re = Reynolds number
			Pr = Prandtl number
			rpi = inner radius pipe
			H = length of BHE
	'''
	if Re < 2300:
		return 4.364		
	elif Re >= 10000:
		Xir = (1.8*np.log10(Re)-1.5)**(-2)
		return (Xir/8*Re*Pr)/(1+12.7*(Xir/8)**0.5 *(Pr**(2/3)-1)) * (1+(2*rpi/(2*H))**(2/3))		
	else:
		return (1-(Re-2300)/(10000-2300))*4.364 +(Re-2300)/(10000-2300)*((0.0308/8*10000*Pr)/(1+12.7*(0.0308/8)**0.5*(Pr**(2/3)-1))*(1+ (2*rpi/(2*H))**(2/3)))        

def Nusselt1U (Re,Pr,rpi,H):
	'''
	calculate Nusselt Number for 1U BHE
	inputs: Re = Reynolds number
			Pr = Prandtl number
			rpi = inner radius pipe
			H = length of BHE
	'''
	if Re < 2300:
		return 4.364		
	elif Re >= 10000:
		Xir = (1.8*np.log10(Re)-1.5)**(-2)
		return (Xir/8*Re*Pr)/(1+12.7*(Xir/8)**0.5 *(Pr**(2/3)-1)) * (1+(2*rpi/(H))**(2/3))		
	else:
		return (1-(Re-2300)/(10000-2300))*4.364 +(Re-2300)/(10000-2300)*((0.0308/8*10000*Pr)/(1+12.7*(0.0308/8)**0.5*(Pr**(2/3)-1))*(1+ (2*rpi/(H))**(2/3)))   

def NusseltCoaxo (Reo,Pr,ro_i,H):
	'''
	calculate Nusselt Number for inner pipe of coaxial BHE
	inputs: Reo = Reynolds number
			Pr = Prandtl number
			ro_i = inner radius pipe
			H = length of BHE
	'''
	if Reo < 2300:
		#print ("laminar")
		return 4.364		
	elif Reo >= 10000:
		Xir = (1.8*np.log10(Reo)-1.5)**(-2)
		#print ("turbulent")
		return (Xir/8*Reo*Pr)/(1+12.7*(Xir/8)**0.5 *(Pr**(2/3)-1)) * (1+(2*ro_i/(H))**(2/3))		
	else:
		#print ("turbulent")
		return (1-(Reo-2300)/(10000-2300))*4.364 + (Reo-2300)/(10000-2300)*((0.0308/8*10000*Pr)/(1+12.7*(0.0308/8)**0.5*(Pr**(2/3)-1))*(1+ (2*ro_i/(H))**(2/3)))        

def NusseltCoaxi (Rei,Pr,do,di_i,dh,H):
	'''
	calculate Nusselt Number for outer pipe of coaxial BHE
	inputs: Rei = Reynolds number
			Pr = Prandtl number
			do = outer diameter inner pipe
			di_i = inner diameter outer pipe
			dh = di_i - do
			H = length of BHE
	'''
	if Rei < 2300:
		#print ("laminar")
		return 3.66+(4-0.102/((do/di_i)+0.02))*(do/di_i)**0.04
	elif Rei >= 10000:
		#print ("turbulent")
		Xir = (1.8*np.log10(Rei)-1.5)**(-2)
		return (Xir/8*Rei*Pr)/(1+12.7*(Xir/8)**0.5 *(Pr**(2/3)-1)) * (1+ (dh/(H))**(2/3)) * (0.86*(do/di_i)**0.84 + 1-0.14*(do/di_i)**0.6)/(1+do/di_i)  
	else:
		#print ("uebergang")
		return (1-(Rei-2300)/(10000-2300)) * (3.66+(4-0.102/((do/di_i)+0.02))*(do/di_i)**0.04) + (Rei-2300)/(10000-2300) * ((0.0308/8*10000*Pr)/(1+12.7*(0.0308/8)**0.5*(Pr**(2/3)-1))*(1+ (dh/(H))**(2/3)))* ((0.86*(do/di_i)**0.84 + 1-0.14*(do/di_i)**0.6)/( 1+do/di_i))
	
	
