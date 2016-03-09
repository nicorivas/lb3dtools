# Python module to visualize lb3d data in python
# Uses matplotlib extensively

import os
import sys
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lb3dpytools.sims

#==============================================================================
def pdensity(field, vmin=-1, vmax=-1, colorbar=0):
	'''Plots a colour density plot for a 2d field.
	Uses pcolormesh and sets the proper color limits and ranges.
		@field: numpy 2d array of data to plot
		@vmin: sets minimum value of color scale
		@vmax: sets maximum value of color scale
		@colorbar: if to show colorbar
		returns: matplotlib.collections.QuadMesh object
	'''
	ly = len(field)
	lx = len(field[0])
	
	# Determine color range
	if (vmin == vmax): # if it was not set outside, or just set stupidly
		vmin = np.min(field)
		vmax = np.max(field)
	
	p = plt.pcolormesh(
			np.arange(0,ly+1,1),
			np.arange(0,lx+1,1),
			field.T,
			vmin=vmin,vmax=vmax)
	
	if colorbar:
		plt.colorbar(p)
	
	# This assumes coordinates start at 0. But it is always like that in lb3d
	plt.xlim(0,ly)
	plt.ylim(0,lx)

	return p

#==============================================================================
def pprofile(field, axes=['x','y'], label='', title=''):
	'''Plots a x-y plot for a 1d field cut
	'''
	if len(_field.shape) != 1:
		print("WARNING: wrong dimensionality of field, aborting plot")
		return

	lx = len(field)

	step = 1.0
	xs = np.arange(0+step/2.0,lx,step)
	ys = cut

	p = plt.plot(xs, ys, marker='o', label=label)
	plt.title(title)
	plt.xlabel(axes[0])
	plt.ylabel(axes[1])
	plt.xlim(0,lx)

	return p

#==============================================================================
def pprofiles():

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
def pcolloid():
	xc = 10.0
	r = 4.5
	xi = xc - r/2.0
	yi = 0.0
	axes = plt.gca()
	axes.add_patch(patches.Rectangle((xi,yi),r*2,10.0))

#==============================================================================
# TODO This should be moved
def _getColloidTrajectory(_prefix, _dir):
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
# TODO This shouldn't be here
def partition(_field, _length):
	"""Given an array, returns sub arrays of length _length
	"""
	return [_field[i:i+_length] for i in range(0, len(_field), _length)]
