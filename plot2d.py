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

#==============================================================================
def plotDensity(field, vmin=-1, vmax=-1, colorbar=0):
	'''
	Plots a colour density plot for a 2d field.
	Uses pcolormesh and sets the proper color limits and ranges.
	'''
	cut = field
	ly = len(cut)
	lx = len(cut[0])
	if (vmin == vmax): # if it was not set outside, or just set stupidly
		vmin = np.min(cut)
		vmax = np.max(cut)
	p = plt.pcolormesh(np.arange(0,ly+1,1), np.arange(0,lx+1,1), cut.T, 
			vmin=vmin,vmax=vmax)
	if colorbar:
		plt.colorbar(p)
	plt.xlim(0,ly)
	plt.ylim(0,lx)

#==============================================================================
def plotProfile(_field, axes=['x','y'],label='',title=''):
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

	plt.plot(xs, ys, marker='o', label=label)
	plt.title(title)
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

#==============================================================================
# TODO This shouldn't be here
def partition(_field, _length):
	"""Given an array, returns sub arrays of length _length
	"""
	return [_field[i:i+_length] for i in range(0, len(_field), _length)]
