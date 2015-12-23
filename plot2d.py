# Module to visualize lb3d data in python, using matplotlib.

import os
import sys
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('/data/home/nicorivas/Code/analysis/lb3d/')
import lb_sims

G_sim = None # Current simulation being worked on

#==============================================================================
# TODO This shouldn't be here
def partition(_field, _length):
	"""Given an array, returns sub arrays of length _length
	"""
	return [_field[i:i+_length] for i in range(0, len(_field), _length)]

#==============================================================================
def setSimulation(_sim):
	"""Define current simulation used by all plotting functions.
	"""
	global G_sim
	G_sim = _sim

#==============================================================================
def setFigureSize(_x, _y):
	#fig = plt.gcf()
	#fig.set_size_inches(30.0, 30.0)
	plt.figure(figsize=(_x, _y))


#==============================================================================
def plotDensity(_field,_vmin=-1,_vmax=-1):
	'''Plots a colour density plot for a 2d field cut
	'''
	cut = _field
	lx = len(cut)
	ly = len(cut[0])
	if (_vmin == _vmax): # if it was not set outside, or just set stupiddly
		_vmin = np.min(cut)
		_vmax = np.max(cut)
	plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut,
			vmin=_vmin,vmax=_vmax)
	plt.xlim(0,lx)
	plt.ylim(0,ly)

#==============================================================================
def plotProfile(_field, axes=['x','y']):
	'''Plots a x-y plot for a 1d field cut
	'''
	if len(_field.shape) != 1:
		print("WARNING: wrong dimensionality of field, aborting plot")
		return
	cut = _field
	lx = len(cut)

	step = 1.0
	xs = np.arange(0+step/2.0,lx,step)
	ys = cut

	plt.plot(xs, ys)
	plt.xlabel(axes[0])
	plt.ylabel(axes[1])
	plt.xlim(0,lx)

#==============================================================================
def plotProfiles():

	fig = plt.figure()
	axes = plt.axes()

	profiles = getProfiles('profile-x_','Q_wall0.1',1)
	plotProfile(fig, axes, profiles[:,(0,3)])

	# Theoretical plot
	L = 20.0
	lambda_B = 0.4
	e = 1.0
	E = 0.025
	eta = 1.0/6.0
	K = 0.02766

	v0 = (e*E)/(eta*2.0*math.pi*lambda_B)
	xs = np.arange(0,L,0.1)
	ys = [v0*np.log((np.cos(K*(x-0.5*L)))/np.cos(0.5*K*L)) for x in xs]
	axes.plot(xs,ys)

	plt.show()

#==============================================================================
def plotColloid1D():
	xc = 10.0
	r = 4.5
	xi = xc - r/2.0
	yi = 0.0
	axes = plt.gca()
	axes.add_patch(patches.Rectangle((xi,yi),r*2,10.0))

#==============================================================================
# TODO This should be moved
def getColloidTrajectory(_prefix, _dir):
	"""Returns a list of the particles' position x,y,z in time
	"""
	trajectory = []
	filenamesmd = glob.glob(_dir+'/'+_prefix+'*')
	for fn in filenamesmd:
		file = open(fn)
		for line in file:
			lp = line.split()
			lp = [float(x) for x in lp]
			trajectory.append([lp[0],lp[1],lp[2]])
	return trajectory

#==============================================================================
def plotFields():
	
	global eps, phi, rhom, rhop
	x = 15	
	plt.subplot(2,2,1)
	cut = eps[x,:,:]
	lx = len(cut)
	ly = len(cut[0])
	plt.xlabel('y')
	plt.ylabel('z')
	plt.title('eps')
	plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut)
	
	plt.subplot(2,2,2)
	cut = phi[x,:,:]
	lx = len(cut)
	ly = len(cut[0])
	plt.xlabel('y')
	plt.ylabel('z')
	plt.title('phi')
	plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut)
	
	plt.subplot(2,2,3)
	cut = rhom[x,:,:]
	lx = len(cut)
	ly = len(cut[0])
	plt.xlabel('y')
	plt.ylabel('z')
	plt.title('rho m')
	plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut)
	
	plt.subplot(2,2,4)
	cut = rhop[x,:,:]
	lx = len(cut)
	ly = len(cut[0])
	plt.xlabel('y')
	plt.ylabel('z')
	plt.title('rho p')
	plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut)
	
	plt.tight_layout(pad=0.5)
	plt.show()

# Gets data from files of 2D slices of 3D data
def getProfiles(_prefix, _dir, _timeFileIndex=0):
	# _timeFileIndex is the index in the file list that we obtain through glob.
	# it usually corresponds to a particular time-step.
	nc = 11 # Number of fields, TODO: Hardcoded
	f_all = glob.glob('output/'+_dir+'/'+_prefix+'*.asc')
	file = open(f_all[_timeFileIndex], 'r')
	i = 0
	for line in file:
		if line[0] == '#':
			# We pray the lord that comments are on the top of the file
			if line.find("profile length") > -1:
				pl = float(line[line.find(":")+1:-1])
				profiles = np.zeros((pl,nc))
		else:
			# This is data, (and nothing else, please).
			# Fortran exports some numbers without the E letter, and Python complains
			# when converting to float. The lambda map fixes that. TODO: very slow
			ls = line.split()
			ls = map(lambda x: 0 if (x.find('-') > -1 and x.find('E') == -1) else x,ls)
			vals = [float(n) for n in ls]
			profiles[i] = vals #< copying sequence to array here (slow)
			i += 1
	return profiles

def plotProfilesFromFields():
	
	global eps, phi, rhom, rhop
	
	plt.subplot(2,2,1)
	size = eps.shape
	plt.xlabel('x')
	plt.ylabel('z')
	plt.title('eps')
	for y in np.arange(0,size[1],math.floor(size[1]/5)):
		yi = int(y)
		plt.plot(np.arange(0,size[0],1), eps[0,yi,:])
		plt.plot(np.arange(0,size[0],1), eps[0,yi,:])
		plt.plot(np.arange(0,size[0],1), eps[0,yi,:])
		plt.plot(np.arange(0,size[0],1), eps[0,yi,:])
		plt.plot(np.arange(0,size[0],1), eps[0,yi,:])
	
	plt.subplot(2,2,2)
	size = phi.shape
	plt.xlabel('x')
	plt.ylabel('z')
	plt.title('phi')
	for y in np.arange(0,size[1],math.floor(size[1]/5)):
		yi = int(y)
		plt.plot(np.arange(0,size[0],1), phi[0,yi,:])
		plt.plot(np.arange(0,size[0],1), phi[0,yi,:])
		plt.plot(np.arange(0,size[0],1), phi[0,yi,:])
		plt.plot(np.arange(0,size[0],1), phi[0,yi,:])
		plt.plot(np.arange(0,size[0],1), phi[0,yi,:])
	
	plt.subplot(2,2,3)
	size = rhom.shape
	plt.xlabel('x')
	plt.ylabel('z')
	plt.title('rhom')
	for y in np.arange(0,size[1],math.floor(size[1]/5)):
		yi = int(y)
		plt.plot(np.arange(0,size[0],1), rhom[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhom[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhom[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhom[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhom[0,yi,:])
	
	plt.subplot(2,2,4)
	size = rhop.shape
	plt.xlabel('x')
	plt.ylabel('z')
	plt.title('rhop')
	for y in np.arange(0,size[1],math.floor(size[1]/5)):
		yi = int(y)
		plt.plot(np.arange(0,size[0],1), rhop[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhop[0,yi,:])
		plt.plot(np.arange(0,size[0],1), rhop[0,yi,:])
	
	plt.tight_layout(pad=0.5)
	plt.show()

"""
def getElecFields(_prefix, _dir, _format, _n=0):
	'''Gets data from files with 3D matrixes of the different fields.
	'''
	if _format == 'hdf5':
		f_eps = glob.glob(_dir+'/'+_prefix+'eps*')
		f_phi = glob.glob(_dir+'/'+_prefix+'phi*')
		f_rhom = glob.glob(_dir+'/'+_prefix+'rho_m*')
		f_rhop = glob.glob(_dir+'/'+_prefix+'rho_p*')
	
		if len(f_eps) == 0:
			print("File not found")
			exit(1)

		file_eps = h5py.File(f_eps[_n],'r')
		eps = np.array(file_eps['OutArray'])
		file_phi = h5py.File(f_phi[_n],'r')
		phi = np.array(file_phi['OutArray'])
		file_rhom = h5py.File(f_rhom[_n],'r')
		rhom = np.array(file_rhom['OutArray'])
		file_rhop = h5py.File(f_rhop[_n],'r')
		rhop = np.array(file_rhop['OutArray'])
		return [eps, phi, rhom, rhop]
	elif _format == 'ascii': #ascii output files, see format in a file.
		#TODO: THIS HAS NEVER BEEN TESTED!
		f_all = glob.glob(_dir+'/'+_prefix+'*.asc')
		if len(f_all) == 0:
			print("No output files found, looking for ascii")
			exit(1)
		file = open(f_all[0],'r')
		
		rho_p = np.zeros((_nx, _ny, _nz))
		rho_m = np.zeros((_nx, _ny, _nz))
		phi = np.zeros((_nx, _ny, _nz))
		eps = np.zeros((_nx, _ny, _nz))
		ex = np.zeros((_nx, _ny, _nz))
		ey = np.zeros((_nx, _ny, _nz))
		ez = np.zeros((_nx, _ny, _nz))
		op = np.zeros((_nx, _ny, _nz))
		rock = np.zeros((_nx, _ny, _nz))
		for l in file:
			if l[0] != '#':
				vals = [float(n) for n in l.split()]
				rho_p[int(vals[0])][int(vals[1])][int(vals[2])] = vals[3]
				rho_m[int(vals[0])][int(vals[1])][int(vals[2])] = vals[4]
				phi[int(vals[0])][int(vals[1])][int(vals[2])] = vals[5]
				eps[int(vals[0])][int(vals[1])][int(vals[2])] = vals[6]
				ex[int(vals[0])][int(vals[1])][int(vals[2])] = vals[7]
				ey[int(vals[0])][int(vals[1])][int(vals[2])] = vals[8]
				ez[int(vals[0])][int(vals[1])][int(vals[2])] = vals[9]
				op[int(vals[0])][int(vals[1])][int(vals[2])] = vals[10]
				rock[int(vals[0])][int(vals[1])][int(vals[2])] = vals[11]
	elif _format == 'bz2':
		f_all = glob.glob(_dir+'/'+'*.bz2')
		
		if len(f_all) == 0:
			print('No output files found, looking for .bz2')
			exit(1)
		
		x = []
		y = []
		z = []
		phi = []
		rhop = []
		rhom = []
		
		x1, y1, z1, phi1, rho_p1, rho_m1 = np.loadtxt(f_all[1], unpack=True, usecols=[0,1,2,7,11,12])
		
		x.extend(x1)
		y.extend(y1)
		z.extend(z1)
		phi.extend(phi1)
		rhop.extend(rho_p1)
		rhom.extend(rho_m1)
		
		phi = np.asarray(partition(partition(phi,30),30))
		rhom = np.asarray(partition(partition(rhom,30),30))
		rhop = np.asarray(partition(partition(rhop,30),30))

		return [phi, rhom, rhop]

"""
