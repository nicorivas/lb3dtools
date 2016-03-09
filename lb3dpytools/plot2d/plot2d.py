#!/usr/bin/env python
import sys, subprocess, glob, os, argparse
import h5py
import math
import numpy as np
#sys.path.append('/usr/lib/pymodules/python2.7/') #because the install in SUN is wierd
import matplotlib.pyplot as plt
#sys.path.append('/data/home/nicorivas/Code/analysis/lb3d/')
import core
import lb3dtools.sims

# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================

# These are level 2 functions: they should use the level 1 functions only,
# and they are pertinent only for lb3d simulations, that is, they label and
# make assumptions based on the data from lb3d.

def plot_velprofile(sim, time=0, fieldprefix='', dir=0, verbose=0, show=0):
	"""
	1D profiles of velocity
	"""
	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	vx = sim.getField(fieldprefix+'velx', t=time)
	vy = sim.getField(fieldprefix+'vely', t=time)
	vz = sim.getField(fieldprefix+'velz', t=time)

	if isinstance(vx, int) or isinstance(vy, int) or isinstance(vz, int):
		error('Field not found')
		exit(0)

	vxmin = np.min(vx)
	vxmax = np.max(vx)
	vymin = np.min(vy)
	vymax = np.max(vy)
	vzmin = np.min(vz)
	vzmax = np.max(vz)

	if verbose:
		print 'min(vx)={} max(vx)={}'.format(vxmin,vxmax)
		print 'min(vy)={} max(vy)={}'.format(vymin,vymax)
		print 'min(vz)={} max(vz)={}'.format(vzmin,vzmax)

	zeros = [False, False, False]
	if vxmin == 0.0 and vxmax == 0.0:
		zeros[0] = True
	if vymin == 0.0 and vymax == 0.0:
		zeros[1] = True
	if vzmin == 0.0 and vzmax == 0.0:
		zeros[2] = True

	if zeros[0] and zeros[1] and zeros[2]:
		print('The field is zero, refusing to plot')
		return 1

	'''
	velx_y = (vx.mean(axis=2)).mean(axis=0)
	vely_y = (vy.mean(axis=2)).mean(axis=0)
	velz_y = (vz.mean(axis=2)).mean(axis=0)
	plt.plot(velx_y,label='velx')
	plt.plot(vely_y,label='vely')
	plt.plot(velz_y,label='velz')
	plt.xlabel('y')
	plt.legend(loc='best')
	'''
	velx_xy = vx.mean(axis=2)
	vely_xy = vy.mean(axis=2)
	velz_xy = vz.mean(axis=2)
	#for i in range(0, nx):
	for i in range(0, 1):
		#plt.plot(velx_xy[:,i], label='velx')
		#plt.plot(vely_xy[:,i], label='vely')
		#plt.plot(velz_xy[:,i], label='velz')
		plt.plot(vx[:,i,0], label='velx')
		plt.plot(vy[:,i,0], label='vely')
		plt.plot(vz[:,i,0], label='velz')
		print(vx)
		print(vy)
		print(vz)
		plt.xlim([0-1,nx+1])
		plt.xlabel('x')
	
	plt.legend(loc='best')

	plt.title('Ex = {} Ey = {}'.format(sim.get(1,'Ex' ),sim.get(1,'Ez')))

	if show:
		plt.show()
	
	'''
	velx_y = vx.mean(axis=2)
	vely_y = vy.mean(axis=2)
	velz_y = vz.mean(axis=2)
	plt.subplot(1,2,1)
	plot2d.plotDensity(velx_y)
	plt.subplot(1,2,2)
	plot2d.plotDensity(vely_y)
	'''

	#if dir==0:
	#vs = vx*vx+vy*vy+vz*vz
	#print(np.max(vs))
	#vp = [np.mean(v) for v in vs]
	#print(vp)
	#plt.plot(vp)


def plot_veldensity(sim, time=0, fieldprefix='', dir=0, verbose=0):
	"""
	Density plots of the norm of the velocity
	"""
	sim.loadOutput()
	vx = sim.getField(fieldprefix+'velx', t=time)
	vy = sim.getField(fieldprefix+'vely', t=time)
	vz = sim.getField(fieldprefix+'velz', t=time)

	if isinstance(vx, int) or isinstance(vy, int) or isinstance(vz, int):
		error('Field not found')
		exit(0)

	vxmin = np.min(vx)
	vxmax = np.max(vx)
	vymin = np.min(vy)
	vymax = np.max(vy)
	vzmin = np.min(vz)
	vzmax = np.max(vz)

	if verbose:
		print 'min(vx)={} max(vx)={}'.format(vxmin,vxmax)
		print 'min(vy)={} max(vy)={}'.format(vymin,vymax)
		print 'min(vz)={} max(vz)={}'.format(vzmin,vzmax)

	zeros = [False, False, False]
	if vxmin == 0.0 and vxmax == 0.0:
		zeros[0] = True
	if vymin == 0.0 and vymax == 0.0:
		zeros[1] = True
	if vzmin == 0.0 and vzmax == 0.0:
		zeros[2] = True

	if zeros[0] and zeros[1] and zeros[2]:
		print('The field is zero, refusing to plot')
		return 1


	#if dir==0:
	vs = vx*vx+vy*vy+vz*vz
	print(np.max(vs))
	print(vs)
	ct_scan(sim, vs, dir=0)

def plot_velfield(sim, time=0, fieldprefix='', dir=0, cut=0, show=0, verbose=0):
	"""
	Plots a velocity field in 2D. Only components parallel to one of the
	ortonormal planes are plotted, with alternatively color for the other dir.
		@dir: direction normal to the plane of the vector field
		@cut: index of where to make the cut in the direction dir
	"""

	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	vx = sim.getField(fieldprefix+'velx', t=time)
	vy = sim.getField(fieldprefix+'vely', t=time)
	vz = sim.getField(fieldprefix+'velz', t=time)

	if isinstance(vx, int) or isinstance(vy, int) or isinstance(vz, int):
		error('Field not found')
		exit(0)

	vxmin = np.min(vx)
	vxmax = np.max(vx)
	vymin = np.min(vy)
	vymax = np.max(vy)
	vzmin = np.min(vz)
	vzmax = np.max(vz)

	if verbose:
		print 'min(vx)={} max(vx)={}'.format(vxmin,vxmax)
		print 'min(vy)={} max(vy)={}'.format(vymin,vymax)
		print 'min(vz)={} max(vz)={}'.format(vzmin,vzmax)

	zeros = [False, False, False]
	if vxmin == 0.0 and vxmax == 0.0:
		zeros[0] = True
	if vymin == 0.0 and vymax == 0.0:
		zeros[1] = True
	if vzmin == 0.0 and vzmax == 0.0:
		zeros[2] = True

	if zeros[0] and zeros[1] and zeros[2]:
		print('The field is zero, refusing to plot')
		return 1

	lx = nx
	ly = ny
	labx = 'x'
	laby = 'y'
	if dir == 0:
		dirs = 'x'
		fx = vy[cut,:,:]
		fy = vz[cut,:,:]
		fz = vx[cut,:,:]
		lx = ny
		ly = nz
		labx = 'y'
		laby = 'z'
	elif dir == 1:
		dirs = 'y'
		fx = vx[:,cut,:]
		fy = vz[:,cut,:]
		fz = vy[:,cut,:]
		lx = nx
		ly = nz
		labx = 'x'
		laby = 'z'
	elif dir == 2:
		dirs = 'z'
		fx = vx[:,:,cut]
		fy = vy[:,:,cut]
		fz = vz[:,:,cut]
		lx = nx
		ly = ny
		labx = 'x'
		laby = 'y'

	# color:
	#cs = np.hypot(fx, fy) # according to norm of in-plane vectors
	cs = fz # according to (signed) norm of out of plane component
	

	sh = 0.5
	xs, ys = np.meshgrid(
			np.arange(0.0+sh,lx+sh,1),
			np.arange(0.0+sh,ly+sh,1))

	#print(fx)
	#print(fy)
	#fx = [max(10,f) for f in [a for a in fx]]
	#fy = [max(10,f) for f in [a for a in fy]]

	res = 1 # skip every 'res' values
	#c = plt.quiver(xs[::res,::res], ys[::res,::res], vy[::res,::res],
	#		vz[::res,::res], cs[::res,::res])
	c = plt.quiver(xs[::res,::res], ys[::res,::res], fx[::res,::res],
			fy[::res,::res],cs[::res,::res])
	plt.xlabel(labx)
	plt.ylabel(laby)
	plt.xlim(0, lx)
	plt.ylim(0, ly)
	plt.title('time={} {}={}'.format(time, dirs, cut))

	if show:
		plt.show()


# Plot potential ==============================================================
def plot_potential(sim, dir=0, time=0, fieldname='elec-', show=0, cut=0, verbose=1):
	sim.loadOutput()

	phi = sim.getField(fieldname+'phi',t=time)

	phimin = np.min(phi)
	phimax = np.max(phi)

	if verbose:
		print 'min(phi)={} max(phi)={}'.format(phimin,phimax)
	if phimin == 0.0 and phimax == 0.0:
		print('The field appears to be zero, refusing to plot')
		return 1
	if phimin == phimax:
		print('Field appears to be homogeneous')
	
	print(phimin)
	print(phimax)
	
	#ct_scan(sim, phi, dir=0, title='phi')
	cut = 8
	#print(phi)
	rho_m = phi[:,8,:]
	print(rho_m)
	plot2d.plotDensity(rho_m,colorbar=1, vmin=.0006, vmax=0.0)
	plt.tight_layout()

	if show:
		plt.show()

# Plot dielectric field =======================================================
def plot_dielectric(sim, dir=0, time=0, fieldname='elec-', show=0, cut=0):
	sim.loadOutput()
	eps = sim.getField(fieldname+'eps')

	epsmin = np.min(eps)
	epsmax = np.max(eps)

	print 'min(eps)={} max(eps)={}'.format(epsmin,epsmax)
	if epsmin == 0.0 and epsmax == 0.0:
		print('The field appears to be zero, refusing to plot')
		return 1
	if epsmin == epsmax:
		print('Field appears to be homogeneous')
	
	ct_scan(sim, eps, dir=0, title='eps')
	plt.tight_layout()
	plt.show()

# Plot charge field ===========================================================
def plot_charge_profile(sim, time=0, fieldprefix='elec-', dir=0, verbose=0,
		cut=[0,0], show=0):
	"""
	Given a simulation object, print the charge field (both rho_m and rho_p),
	as a profile in a given direction, for given cuts.
	"""
	sim.loadOutput()

	fieldname = fieldprefix+'rho_m'
	rho_m = sim.getField(fieldname,t=time)
	if isinstance(rho_m, int):
		error("Field not found '{}' at time '{}'".format(fieldname, time))
		exit(0)

	fieldname = fieldprefix+'rho_p'
	rho_p = sim.getField(fieldname,t=time)
	if isinstance(rho_p, int):
		error("Field not found '{}' at time '{}'".format(fieldname, time))
		exit(0)

	rhommin = np.min(rho_m)
	rhommax = np.max(rho_m)
	rhopmin = np.min(rho_p)
	rhopmax = np.max(rho_p)

	if verbose:
		print 'min(rho_m)={} max(rho_m)={}'.format(rhommin,rhommax)
		print 'min(rho_p)={} max(rho_p)={}'.format(rhopmin,rhopmax)

	zeros = [False, False]
	if rhommin == 0.0 and rhommax == 0.0:
		zeros[0] = True
	if rhopmin == 0.0 and rhopmax == 0.0:
		zeros[1] = True
	if zeros[0] and zeros[1]:
		message('The field appears to be zero (rhopmin = {}, rhopmax = {}),'
		' refusing to plot'.format(rhopmin,rhopmax))
		return 1

	if dir==0:
		rho_m = rho_m[:,cut[0],cut[1]]
		rho_p = rho_p[:,cut[0],cut[1]]
		xl = 'x'
	elif dir==1:
		rho_m = rho_m[cut[0],:,cut[1]]
		rho_p = rho_p[cut[0],:,cut[1]]
		xl = 'y'
	elif dir==2:
		rho_m = rho_m[cut[0],cut[1],:]
		rho_p = rho_p[cut[0],cut[1],:]
		xl = 'z'

	plt.plot(rho_m)
	plt.xlim([0,16])
	plt.ylim([0.039,0.04])
	#plt.plot(rho_p)
	#plt.xlabel(xl)
	#plt.ylabel('rho')
	#plt.title('rho_m, rho_p')
	#plt.tight_layout()

	if show:
		plt.show()

# Plot charge profiles ========================================================
def plot_charge(sim, time=0, fieldprefix='elec-', dir=0, verbose=0,
		cut=0, show=0):
	"""
	Given a simulation object, print the charge field (both rho_m and rho_p),
	as individual large density fields at position 'cut', and ct_scans, all
	in direction 'dir'.
	"""
	sim.loadOutput()

	fieldname = fieldprefix+'rho_m'
	rho_m = sim.getField(fieldname,t=time)
	if isinstance(rho_m, int):
		error("Field not found '{}' at time '{}'".format(fieldname, time))
		exit(0)

	fieldname = fieldprefix+'rho_p'
	rho_p = sim.getField(fieldname,t=time)
	if isinstance(rho_p, int):
		error("Field not found '{}' at time '{}'".format(fieldname, time))
		exit(0)

	rhommin = np.min(rho_m)
	rhommax = np.max(rho_m)
	rhopmin = np.min(rho_p)
	rhopmax = np.max(rho_p)

	if verbose:
		print 'min(rho_m)={} max(rho_m)={}'.format(rhommin,rhommax)
		print 'min(rho_p)={} max(rho_p)={}'.format(rhopmin,rhopmax)

	zeros = [False, False]
	if rhommin == 0.0 and rhommax == 0.0:
		zeros[0] = True
	if rhopmin == 0.0 and rhopmax == 0.0:
		zeros[1] = True
	if zeros[0] and zeros[1]:
		message('The field appears to be zero (rhopmin = {}, rhopmax = {}),'
		' refusing to plot'.format(rhopmin,rhopmax))
		return 1

	#ct_scan(sim, rho_m, dir=dir, title='rho_m')
	#ct_scan(sim, rho_p, dir=dir, title='rho_p')

	if dir==0:
		rho_m = rho_m[cut,:,:]
		rho_p = rho_p[cut,:,:]
		xl = 'y'
		yl = 'z'
	elif dir==1:
		rho_m = rho_m[:,cut,:]
		rho_p = rho_p[:,cut,:]
		xl = 'x'
		yl = 'z'
	elif dir==2:
		rho_m = rho_m[:,:,cut]
		rho_p = rho_p[:,:,cut]
		xl = 'x'
		yl = 'y'

	#plt.figure(figsize=[12,6])
	#plt.subplot(1,2,1)
	#plot2d.plotDensity(rho_m, colorbar=1)#
	plot2d.plotDensity(rho_p, colorbar=1)#, vmin=0.03, vmax=0.2)
	#plt.xlabel(xl)
	#plt.ylabel(yl)
	#plt.title('rho_m')
	#plt.tight_layout()

	#plt.subplot(1,2,2)
	#plot2d.plotDensity(rho_p)
	#plt.xlabel(xl)
	#plt.ylabel(yl)
	#plt.title('rho_p')
	#plt.tight_layout()

	if show:
		plt.show()

# Plot density fields =========================================================
def plot_density(sim, time=0, fieldname='elec-', show=1, verbose=0):
	sim.loadOutput()
	od = sim.getField('od',t=time)
	wd = sim.getField('wd',t=time)

	odmin = np.min(od)
	odmax = np.max(od)
	wdmin = np.min(wd)
	wdmax = np.max(wd)

	if verbose:
		print 'min(od)={} max(od)={}'.format(odmin,odmax)
		print 'min(wd)={} max(wd)={}'.format(wdmin,wdmax)

	zeros = [False, False]
	if odmin == 0.0 and odmax == 0.0:
		zeros[0] = True
	if wdmin == 0.0 and wdmax == 0.0:
		zeros[1] = True
	if zeros[0] and zeros[1]:
		print('The field appears to be zero, refusing to plot')
		return 1

	ct_scan(sim, od, dir=0)
	ct_scan(sim, wd, dir=0)

	z = 0
	od = od[z,:,:]
	wd = wd[z,:,:]

	plt.figure(figsize=[12,6])
	plt.subplot(1,2,1)
	plot2d.plotDensity(od)
	plt.title('od (time={} z={})'.format(time, z))
	plt.tight_layout()

	plt.subplot(1,2,2)
	plot2d.plotDensity(wd)
	plt.title('wd (time={} z={})'.format(time, z))
	plt.tight_layout()
	if show:
		plt.show()

# Plot electric field =========================================================
def plot_efield(sim, time=0, fieldname='elec-', show_density=False, verbose=0):
	sim.loadOutput()
	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	ex = sim.getField(fieldname+'Ex', t=time)
	ey = sim.getField(fieldname+'Ey', t=time)
	ez = sim.getField(fieldname+'Ez', t=time)

	if show_density:
		od = sim.getField('od',t=time)
		wd = sim.getField('wd',t=time)
		phi = (od - wd)/(od + wd)

	exmi = np.min(ex)
	exma = np.max(ex)
	eymi = np.min(ey)
	eyma = np.max(ey)
	ezmi = np.min(ez)
	ezma = np.max(ez)
	if verbose:
		print 'min(ex)={} max(ex)={}'.format(exmi,exma)
		print 'min(ey)={} max(ey)={}'.format(eymi,eyma)
		print 'min(ez)={} max(ez)={}'.format(ezmi,ezma)

	zeros = [False, False, False]
	if exmi == 0.0 and exma == 0.0:
		zeros[0] = True
	if eymi == 0.0 and eyma == 0.0:
		zeros[1] = True
	if ezmi == 0.0 and ezma == 0.0:
		zeros[2] = True

	if zeros[0] and zeros[1] and zeros[2]:
		print('The field is zero, refusing to plot')
		return 1

	ex = ex[0,:,:]
	ey = ey[0,:,:]
	ez = ez[0,:,:]
	fs=20
	plt.figure(figsize=[7,7])
	plt.axes().set_aspect('equal')
	plt.xlim([0.0,ny*1.0])
	plt.ylim([0.0,nz*1.0])
	plt.xlabel('y',fontsize=int(fs/1.0))
	plt.ylabel('z',fontsize=int(fs/1.0))
	plt.xticks(range(0,ny+1,8),fontsize=int(fs/1.2))
	plt.yticks(range(0,nz+1,8),fontsize=int(fs/1.2))
	plt.title(sim.name)
	cs = np.hypot(ey, ez)
	sh = 0.5
	
	if show_density:
		xs, ys = np.meshgrid(
				np.arange(0.0+sh,ny*1.0+sh,1),
				np.arange(0.0+sh,nz*1.0+sh,1))
		print(phi[0,:,:])
		c = plt.pcolormesh(xs, ys, phi[0,:,:].T,cmap='plasma')
		plt.colorbar(c,label='phi',shrink=0.8)
	
	
	xs, ys = np.meshgrid(
			np.arange(0.0+0.5+sh,ny+0.5+sh,1),
			np.arange(0.0+0.5+sh,nz+0.5+sh,1))
	#c = plt.quiver(xs[::2,::2], ys[::2,::2], ey[::2,::2], ez[::2,::2], cs[::2,::2])
	#c = plt.streamplot(xs, ys, ey, ez, density=1.0, color=cs, linewidth=2,arrowsize=2)
	c = plt.streamplot(xs, ys, ey, ez, 
			density=1.0,
			linewidth=cs*7*10*np.max(ez),
			arrowsize=1,
			color=cs,
			cmap='YlGnBu')
	#plt.colorbar(c,label='E',shrink=0.6)
	#plt.show()

# =============================================================================
def plot_localcomposition(sim,time=0):
	sim.loadOutput()
	od = sim.getField('od',t=time)
	wd = sim.getField('wd',t=time)
	if isinstance(od, int) or isinstance(wd, int):
		print('Simulation fields not there')
		exit(0)
	# Definition of local composition field:
	phi = (od - wd)/(od + wd)
	plt.figure(figsize=[12,12])
	ct_scan(sim, phi, dir=0)

# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================
# ============================================================================

# These are level 1 functions: they should use the level 0 (matplotlib)
# functions only, and should work with simulation objects 

def profile(sim, field, dir=0, cuts=[0,0]):
	if dir==0:
		cut = field[:,cuts[0],cuts[1]]
	elif dir == 1:
		cut = field[cuts[0],:,cuts[1]]
	elif dir == 2:
		cut = field[cuts[0],cuts[1],:]
	min = np.min(cut)
	max = np.max(cut)
	plot2d.plotProfile(cut)

def profiles(sim, field, dir=0):

	dir_length_names = ['nx','ny','nz']

	lvls = 30
	nx = int(sim.get(0, 'nx'))-1
	ny = int(sim.get(0, 'ny'))-1
	nz = int(sim.get(0, 'nz'))-1
	dx = 1.0*nx/float(lvls-1)
	dy = 1.0*ny/float(lvls-1)
	dz = 1.0*nz/float(lvls-1)
	w = np.round(np.sqrt((lvls)))
	h = np.ceil(np.sqrt((lvls)))
	max = np.max(field) # for range
	min = np.min(field) # for range

	c = 1
	plot2d.setFigureSize(14.0, 14.0)
	for rx in range(0, lvls):
		if dir == 0:
			x = np.round(rx*dy)
		else:
			x = np.round(rx*dx)
			print(x)
			print(rx)
			print(dx)
		plt.subplot(w,h,c)
		plt.tight_layout()
		plt.ylim([min-(max-min)*0.1,max+(max-min)*0.1])
		#plt.ylim([0.0,0.001])
		for ry in range(0, lvls):
			if dir == 0 or dir == 1:
				y = np.round(ry*dz)
			else:
				y = np.round(ry*dy)
			if dir == 0:
				plot2d.plotProfile(field[:,x,y],axes=['x','phi'],title='y='+str(x),
						label='z='+str(y))
			elif dir == 1:
				plot2d.plotProfile(field[x,:,y],axes=['y','phi'],title='x='+str(x),
						label='z='+str(y))
			elif dir == 2:
				plot2d.plotProfile(field[x,y,:],axes=['z','phi'],title='x='+str(x),
						label='y='+str(y))
		c += 1
		#legend = plt.legend(loc='upper right')
	#plot2d.plotColloid1D()


#------------------------------------------------------------------------------

def ct_scan(sim, field, dir=0, vmin=0, vmax=0, title='', cuts=[]):
	"""
	Series of density plots for cuts in a 3D field in the direction specified by
	dir
	"""

	if dir < 0 or dir > 2:
		error('dir out of bounds')
		exit(0)

	dir_length_names = ['nx','ny','nz']

	plt.figure(figsize=[12.0, 12.0])

	if len(cuts) == 0:
		nl = int(sim.get(0, dir_length_names[dir]))
		values = range(0,nl)
	else:
		values = cuts
		nl = len(values)

	w = np.ceil(np.sqrt(nl))
	h = np.ceil(np.sqrt(nl))

	for c, n in enumerate(values):
		plt.subplot(w,h,c+1)
		if dir == 0:
			plot2d.plotDensity(field[n,:,:],vmin,vmax)
			plt.xlabel('y')
			plt.ylabel('z')
		elif dir == 1:
			plot2d.plotDensity(field[:,n,:],vmin,vmax)
			plt.xlabel('x')
			plt.ylabel('z')
		elif dir == 2:
			plot2d.plotDensity(field[:,:,n],vmin,vmax)
			plt.xlabel('x')
			plt.ylabel('y')
		if title is not '':
			plt.title(title)
		#plt.tight_layout()

#------------------------------------------------------------------------------

def error(string):
	print('(plotter.py) Error! '+string)

def message(string):
	print('(plotter.py) '+string)

def main():

	print("HOLA")
	#==================== FLAG-HANDLING ======================================

	parser = argparse.ArgumentParser(description='ArgumentParser')
	parser.add_argument('--verbose', '-v', action='store_true', help='verbose me please..' )
	parser.add_argument('--plot', '-p', action='store_true', help='plot data because numbers are to complex' )
	args = parser.parse_args()

	# Single simulation plots

	PLOT_DENSITY = False
	PLOT_MULTIPLE_DENSITY = False
	PLOT_CT_SCAN_Z = False
	PLOT_PROFILES_X = False

	project = lb_sims.Project()
	#project.load('ekinetics_singleFluid')
	project.load('ekinetics_droplet')
	project.loadSimulations()
	#sims = ['twoWayCoupling_noForce','twoWayCoupling']
	#sims = ['twoWayCoupling_noColloid']
	#sims = ['test01']

	# Difference of two fields

	sim = project.sims['test01']
	sim.loadOutput()
	field1 = sim.getField('od',t=10)

	#profiles(sim,field1,dir=1)

	#ct_scan(sim, field1, dir=0, vmin=np.min(field1), vmax=np.max(field1))

	sim = project.sims['test02']
	sim.loadOutput()
	field2 = sim.getField('od',t=10)

	#cut = field2[0,:,:]

	#plot2d.plotDensity(cut)

	#plt.show()

	#cd = np.gradient(cut)
	#cd = (cd[0]+cd[1])/2.0

	#plot2d.plotDensity(cd)

	#print(cd)

	#profiles(sim,field2,dir=1)

	#ct_scan(sim, field2, dir=0, vmin=np.min(field2), vmax=np.max(field2))

	field = field2 - field1

	profile(sim,field,cuts=[0,32],dir=1)

	#profiles(sim,field2,dir=2)
	#profiles(sim,field,dir=2)

	#ct_scan(sim, field1, dir=0, vmin=np.min(field1), vmax=np.max(field1))
	#ct_scan(sim, field2, dir=0, vmin=np.min(field2), vmax=np.max(field2))

	#ct_scan(sim, field, dir=1, vmin=np.min(field), vmax=np.max(field))

	plt.show()

if __name__ == "__main__":
	print("HOLA")
	main()
