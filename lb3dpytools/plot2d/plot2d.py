#!/usr/bin/env python
import sys, subprocess, glob, os, argparse
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import core
import lb3dpytools.sims
import lb3dpytools.tools

colors = ['r','g','b','y']*10
say = tools.Say(None)

def mdtrajectory(sim, pn=0, show=0):
	"""Plot the trajectory of a particle in three figures, x-y, x-z and y-z.
		@sim: Simulation object
		@pns: Particle number(s) (int or list)
	"""
	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	if isinstance(pns,int):
		pns = [pns]

	md_data = sim.getMDData()

	plt.figure(figsize=(20,10))
	for pn in enumerate(pns):
		md_datap = md_data[:,pn,:]
		xs = md_datap[:,0]
		ys = md_datap[:,1]
		zs = md_datap[:,2]

		plt.subplot(1,3,1)
		plt.gca().set_aspect('equal')
		plt.plot(xs, ys, color=colors[c])
		plt.xlim(0,nx)
		plt.ylim(0,ny)
		plt.xlabel('x')
		plt.ylabel('y')

		plt.subplot(1,3,2)
		plt.gca().set_aspect('equal')
		plt.plot(xs, zs, color=colors[c])
		plt.xlim(0,nx)
		plt.ylim(0,nz)
		plt.xlabel('x')
		plt.ylabel('z')
	
		plt.subplot(1,3,3)
		plt.gca().set_aspect('equal')
		plt.plot(ys, zs, color=colors[c])
		plt.xlim(0,ny)
		plt.ylim(0,nz)
		plt.xlabel('y')
		plt.ylabel('z')
	
	if show:
		plt.show()

def vel_profile(sim, time=0, fieldprefix='', dir=0, verbose=0, show=0):
	""" 1D profiles of velocity components
	"""
	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	vx = getField(sim, fieldprefix+'velx', time=time)
	vy = getField(sim, fieldprefix+'vely', time=time)
	vz = getField(sim, fieldprefix+'velz', time=time)

	velx_xy = vx.mean(axis=2)
	vely_xy = vy.mean(axis=2)
	velz_xy = vz.mean(axis=2)

	for i in range(0, 1):
		plt.plot(vx[:,i,0], label='velx')
		plt.plot(vy[:,i,0], label='vely')
		plt.plot(vz[:,i,0], label='velz')
		plt.xlim([0-1,nx+1])
		plt.xlabel('x')
	
	plt.legend(loc='best')

	#plt.title('Ex = {} Ey = {}'.format(sim.get(1,'Ex' ),sim.get(1,'Ez')))

	if show:
		plt.show()
	
def vel_density(sim, time=0, fieldprefix='', dir=0, verbose=0, show=0):
	""" CT scan of the norm of the velocity, in direction 'dir'
	"""
	sim.loadOutput()

	vx = getField(sim, fieldprefix+'velx', time=time, verbose=verbose)
	vy = getField(sim, fieldprefix+'vely', time=time, verbose=verbose)
	vz = getField(sim, fieldprefix+'velz', time=time, verbose=verbose)
	
	vs = vx*vx+vy*vy+vz*vz

	ct_scan(sim, vs, dir=dir)

	if show:
		plt.show()

def vel_vector(sim, time=0, fieldprefix='', dir=0, cut=0, show=0, verbose=0):
	""" Plots a velocity field in 2D.
	Only components parallel to one of the ortonormal planes are plotted,
	with alternatively color for the other dir.
		@dir: direction normal to the plane of the vector field
		@cut: index of where to make the cut in the direction dir
	"""
	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	vx = getField(sim, fieldprefix+'velx', time=time, verbose=verbose)
	vy = getField(sim, fieldprefix+'vely', time=time, verbose=verbose)
	vz = getField(sim, fieldprefix+'velz', time=time, verbose=verbose)

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

def epotential(sim, dir=0, time=0, fieldname='elec-', show=0, cut=0,
		vmin=0.0, vmax=0.0, vminp=1.0, vmaxp=1.0, verbose=1):
	""" Plot CT scan of the electric potential in specified direction
		@dir: direction of the CT scan
		@time: time index of file to get
	"""
	sim.loadOutput()

	phi = getField(sim, fieldname+'phi', time=time, verbose=verbose)

	ct_scan(sim, phi, dir=dir, title='phi', vmin=vmin, vmax=vmax, vminp=vminp, vmaxp=vmaxp)

	if show:
		plt.show()

def epotential_density(sim, dir=0, time=0, fieldname='elec-', show=0, cut=0,
		vmin=0.0, vmax=0.0, vminp=1.0, vmaxp=1.0, verbose=1):
	"""Plot CT scan of the electric potential in specified direction
		@dir: direction of the CT scan
		@time: time index of file to get
	"""
	sim.loadOutput()
	phi = getField(sim, fieldname+'phi', time=time, verbose=verbose)
	if dir==0:
		rho_m = phi[cut,:,:]
	elif dir==1:
		rho_m = phi[:,cut,:]
	elif dir==2:
		rho_m = phi[:,:,cut]

	core.pdensity(rho_m,colorbar=1, vmin=vmin, vmax=vmax)

	plt.tight_layout()
	if show:
		plt.show()

def dielectric(sim, dir=0, time=0, fieldname='elec-', show=0, cut=0, verbose=0):
	""" Plot CT scan of the dielectric constant field
	"""
	sim.loadOutput()
	eps = getField(sim, fieldname+'eps', time=time, verbose=verbose)

	ct_scan(sim, eps, dir=0, title='eps')

	plt.tight_layout()
	if show:
		plt.show()

def charge(sim, time=0, fieldprefix='elec-', dir=0, verbose=0,
		cut=0, show=0, vmax=0.0, vmin=0.0, vmaxp=1.0, vminp=1.0):
	"""
	Given a simulation object, print the charge field (both rho_m and rho_p),
	as individual large density fields at position 'cut', and ct_scans, all
	in direction 'dir'.
	"""
	sim.loadOutput()

	fieldname = fieldprefix+'rho_m'
	rho_m = getField(sim, fieldname, time=time, verbose=verbose)
	fieldname = fieldprefix+'rho_p'
	rho_p = getField(sim, fieldname, time=time, verbose=verbose)

	if dir==0:
		rho_m_c = rho_m[cut,:,:]
		rho_p_c = rho_p[cut,:,:]
		xl = 'y'
		yl = 'z'
	elif dir==1:
		rho_m_c = rho_m[:,cut,:]
		rho_p_c = rho_p[:,cut,:]
		xl = 'x'
		yl = 'z'
	elif dir==2:
		rho_m_c = rho_m[:,:,cut]
		rho_p_c = rho_p[:,:,cut]
		xl = 'x'
		yl = 'y'

	ct_scan(sim, rho_p, dir=dir)
	ct_scan(sim, rho_m, dir=dir)

	if show:
		plt.show()

def charge_profile(sim, time=0, fieldprefix='elec-', dir=0, verbose=0,
		cut=[0,0], show=0):
	""" Given a simulation object, print the charge field (both rho_m and rho_p),
	as a profile in a given direction, for given cuts.
	"""
	sim.loadOutput()

	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	fieldname = fieldprefix+'rho_m'
	rho_m = getField(sim, fieldname, time=time, verbose=verbose)
	fieldname = fieldprefix+'rho_p'
	rho_p = getField(sim, fieldname, time=time, verbose=verbose)

	if dir==0:
		rho_m = rho_m[:,cut[0],cut[1]]
		rho_p = rho_p[:,cut[0],cut[1]]
		xl = nx
		xn = 'x'
	elif dir==1:
		rho_m = rho_m[cut[0],:,cut[1]]
		rho_p = rho_p[cut[0],:,cut[1]]
		xl = ny
		xn = 'y'
	elif dir==2:
		rho_m = rho_m[cut[0],cut[1],:]
		rho_p = rho_p[cut[0],cut[1],:]
		xl = nz
		xn = 'z'

	plt.plot(rho_m)
	plt.plot(rho_p)
	plt.xlim([0,xl])
	#plt.plot(rho_p)
	plt.xlabel(xn)
	#plt.ylabel('rho')
	#plt.title('rho_m, rho_p')
	#plt.tight_layout()

	if show:
		plt.show()

def efield(sim, time=0, fieldname='elec-', show_density=False, verbose=0):
	sim.loadOutput()
	nx = int(sim.get(0,'nx'))
	ny = int(sim.get(0,'ny'))
	nz = int(sim.get(0,'nz'))

	ex = getField(sim, fieldname+'Ex', time=time, verbose=verbose)
	ey = getField(sim, fieldname+'Ey', time=time, verbose=verbose)
	ez = getField(sim, fieldname+'Ez', time=time, verbose=verbose)

	if show_density:
		od = getField(sim, 'od', time=time, verbose=verbose)
		wd = getField(sim, 'wd', time=time, verbose=verbose)
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

def density(sim, time=0, dir=0, fieldname='', show=1, verbose=0):
	""" Given a simulation object, print number density fields for every species
	"""
	sim.loadOutput()

	od = getField(sim, 'od', time=time, verbose=verbose)
	wd = getField(sim, 'wd', time=time, verbose=verbose)

	ct_scan(sim, od, dir=dir)
	ct_scan(sim, wd, dir=dir)

	if show:
		plt.show()

def density_density(sim, time=0, dir=0, cut=0, fieldname='', show=1, verbose=0,
		vmin=0.0, vmax=0.0):
	""" Given a simulation object, create density plot of both number density fields
	"""
	sim.loadOutput()

	od = getField(sim, 'od', time=time, verbose=verbose)
	wd = getField(sim, 'wd', time=time, verbose=verbose)

	if dir==0:
		od_c = od[cut,:,:]
		wd_c = wd[cut,:,:]
	elif dir==1:
		od_c = od[:,cut,:]
		wd_c = wd[:,cut,:]
	elif dir==2:
		od_c = od[:,:,cut]
		wd_c = wd[:,:,cut]

	core.pdensity(od_c,colorbar=1, vmin=vmin, vmax=vmax)

	plt.tight_layout()

	if show:
		plt.show()


def localcomposition(sim, time=0, verbose=0, show=0):
	""" Given simulation object, print local composition, (see def. below)
	"""
	sim.loadOutput()

	od = getField(sim, 'od', time=time, verbose=verbose)
	wd = getField(sim, 'wd', time=time, verbose=verbose)
	# Definition of local composition field:
	phi = (od - wd)/(od + wd)

	ct_scan(sim, phi, dir=0)
	if show:
		plt.show()

# ============================================================================

def getField(sim, fieldname, time=0, verbose=0):
	""" Get a field from a simulation, with additional checks for zero or
	homogeneity
	"""
	sim.loadOutput()

	field = sim.getField(fieldname, t=time)

	if isinstance(field, int):
		say.error('Field not found')
		exit(0)

	fmin = np.min(field)
	fmax = np.max(field)
	if verbose:
		say.error('min({})={} max({})={}'.format(fieldname,fmin,fieldname,fmax))
	if fmin == 0.0 and fmax == 0.0:
		say.warning('The field \''+fieldname+'\' is zero!')
	elif fmin == fmax:
		say.warning('The field \''+fieldname+'\' is homogeneous!')

	return field

def ct_scan(sim, field, dir=0, vmin=0, vmax=0, vminp=1.0, vmaxp=1.0, title='', cuts=[]):
	""" Series of density plots of cuts (slices) of a 3D field in the direction
	specified by dir.
	"""
	sim.loadOutput()

	if dir < 0 or dir > 2:
		say.error('\'dir={}\' out of bounds'.format(dir))
		exit(0)

	dir_length_names = ['nx','ny','nz']

	if len(cuts) == 0:
		nl = int(sim.get(0, dir_length_names[dir]))
		values = range(0,nl)
	else:
		values = cuts
		nl = len(values)

	w = np.ceil(np.sqrt(nl))
	h = np.ceil(np.sqrt(nl))

	vmin = np.min(field)*vminp
	vmax = np.max(field)*vmaxp

	plt.figure(figsize=[12.0, 12.0])
	for c, n in enumerate(values):
		plt.subplot(w,h,c+1)
		if dir == 0:
			core.pdensity(field[n,:,:],vmin,vmax)
			plt.xlabel('y')
			plt.ylabel('z')
		elif dir == 1:
			core.pdensity(field[:,n,:],vmin,vmax)
			plt.xlabel('x')
			plt.ylabel('z')
		elif dir == 2:
			core.pdensity(field[:,:,n],vmin,vmax)
			plt.xlabel('x')
			plt.ylabel('y')
		if title is not '':
			plt.title(title)
		#plt.tight_layout()

#=============================================================================

def main():
	""" Main should have a series of tests
	"""
	print("Main funtion is empty")
	exit(0)

if __name__ == "__main__":
	main()
