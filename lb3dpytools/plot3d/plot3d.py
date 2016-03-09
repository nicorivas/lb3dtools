#!/apps/prod/HI-ERN/stow/anaconda2/bin/python

# Module to visualize lb3d data in 3D, using mayavi

import numpy as np
from mayavi import mlab
import vtk
import glob
import h5py
import os.path
import time as tm
import argparse
import re
import lb3danalysis

parser = argparse.ArgumentParser(description='ArgumentParser')
parser.add_argument('--time','-t', action='store', default=-1, type=int, help='Timestep')
parser.add_argument('--duration','-d', action='store', default=10, type=float, help='Duration')
args = parser.parse_args()
count  = 0
time = 0
timesteps = []

files = {}

DATA_PATH = "."

active_membrane = False
active_vel_diff = False
active_water = False
active_oil = False
active_colour = False

def get_velocity_vectors(component='oil', axis='x', step = -1):
	# TODO VALIDATE...
	if component not in ['oil', 'water', 'total'] or axis not in ['x', 'y', 'z']:
		print "Invalid option given to 'get_velocity_vectors'!"
		return

	com = ""
	if component == 'oil':
		com = "od"
	elif component == 'water':
		com = 'wd'
	else:
		print "To be done.."

	a_part = axis
	if step == -1:
		step = time
	files = glob.glob('data/vel%s_%s_out_t%08d-*.h5' % (a_part, comp, step))
	for f in files:
		vectors = read_h5(f)

	return vectors

def add_spurious_currents(start=0, end=100, interval=10, difference=False, vectors=True):
	counter = 0
	files = glob.glob('data/velx_od_out_t%08d-*.h5' % start)
	for f in files:
		velx = read_h5(f)
	u = np.zeros_like(velx)
	v = np.zeros_like(velx)
	w = np.zeros_like(velx)
	for t in np.arange(start, end, interval):
		print "Averaging for timestep %5d" % t
		files = glob.glob('data/velx_od_out_t%08d-*.h5' % t)
		for f in files:
			velx = read_h5(f)
		files = glob.glob('data/vely_od_out_t%08d-*.h5' % t)
		for f in files:
			vely = read_h5(f)
		files = glob.glob('data/velz_od_out_t%08d-*.h5' % t)
		for f in files:
			velz = read_h5(f)
		files = glob.glob('data/velx_wd_out_t%08d-*.h5' % t)
		for f in files:
			velx_w = read_h5(f)
		files = glob.glob('data/vely_wd_out_t%08d-*.h5' % t)
		for f in files:
			vely_w = read_h5(f)
		files = glob.glob('data/velz_wd_out_t%08d-*.h5' % t)
		for f in files:
			velz_w = read_h5(f)
		
		if difference:
			u = u + (velx - velx_w)
			v = v + (vely - vely_w)
			w = w + (velz - velz_w)
		else:
			u = u + (velx + velx_w)
			v = v + (vely + vely_w)
			w = w + (velz + velz_w)
		counter += 1
	
	u = u / counter
	v = v / counter
	w = w / counter
	if difference:
		tit = "Spurious current (vel-diff)"
	else:
		tit = "Spurious current (vel-sum)"
	
	global magn
	if vectors:
		quiver = mlab.pipeline.vector_field(u, v, w)
		qvr = mlab.pipeline.vector_cut_plane(quiver, scale_factor=10, mode='2darrow')
		mlab.vectorbar(qvr, title=tit, orientation="vertical")
	else:
		magn = ( np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w) )**(1./3.)

		xi, yi, zi = np.mgrid[1:60:60j,1:60:60j,1:60:60j]
		grid = mlab.pipeline.scalar_field(xi, yi, zi, magn)
		vol = mlab.pipeline.volume(grid)
		vol.volume_property.shade = False
		#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(magn), plane_orientation='y_axes')
		return magn

def plot_vector_cut(magnitude=False, orientation='y_axes', component=0):
	# Plot vector field widget
	files = glob.glob('data/velx_od_out_t%08d-*.h5' % time)
	for f in files:
		velx = read_h5(f)
	files = glob.glob('data/vely_od_out_t%08d-*.h5' % time)
	for f in files:
		vely = read_h5(f)
	files = glob.glob('data/velz_od_out_t%08d-*.h5' % time)
	for f in files:
		velz = read_h5(f)
	files = glob.glob('data/velx_wd_out_t%08d-*.h5' % time)
	for f in files:
		velx_w = read_h5(f)
	files = glob.glob('data/vely_wd_out_t%08d-*.h5' % time)
	for f in files:
		vely_w = read_h5(f)
	files = glob.glob('data/velz_wd_out_t%08d-*.h5' % time)
	for f in files:
		velz_w = read_h5(f)
	
	if component == 0:  #difference
		u = velx - velx_w
		v = vely - vely_w
		w = velz - velz_w
		title="Velocity difference"
	elif component == 1: # water
		u = velx_w
		v = vely_w
		w = velz_w
		title="Water velocity"
	elif component == 2: # oil
		u = velx
		v = vely
		w = velz
		title="Oil velocity"


	if magnitude:
		mag = ( u**2 + v**2 + w**2 )**(1./3.)
		diff = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(mag), plane_orientation=orientation)
		mlab.colorbar(diff, title=title + "(magnitude)")
	else:
		quiver = mlab.pipeline.vector_field(u, v, w)
		qvr = mlab.pipeline.vector_cut_plane(quiver, scale_factor=10, mode='2darrow')
		mlab.vectorbar(qvr, title=title, orientation="vertical")


def plot_scalar_plane(base="colour_out", orientation='y_axes', title="colour", colorbar=True, vsize=None):
	c_files = glob.glob('data/%s_t%08d-*.h5' % (base, time))
	if c_files == []:
		print "No h5-files found for timestep %d" %  time
		return
	for c in c_files:
		colour = read_h5(c)
		if vsize != None:
			print "Fixing vmax"
			col = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(colour, vmin=vsize[0], vmax=vsize[1], interpolate=False), plane_orientation=orientation)
		else:
			print "AUTO vmax"
			col = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(colour), plane_orientation=orientation)
		if colorbar:
			mlab.colorbar(col, title=title)

""" PLOT VELOCITY VECTORS """ 
def add_velocity_difference():
	global active_vel_diff
	active_vel_diff = True
	plot_vector_cut(magnitude=False, orientation='y_axes', component=0)
def add_water_velocity():
	plot_vector_cut(magnitude=False, orientation='y_axes', component=1)
def add_oil_velocity():
	plot_vector_cut(magnitude=False, orientation='y_axes', component=2)

""" PLOT DENSITY """
def add_colour(orientation='y_axes', volume=False):
	global active_colour
	active_colour = True
	if volume:
		plot_volume(base="colour_out")
	else:
		plot_scalar_plane(base="colour_out",orientation=orientation, title=title)

def add_oil(orientation='y_axes', title="oil-density", volume=False):
	global active_oil
	active_oil = True
	if volume:
		plot_volume(base="od_out")
	else:
		plot_scalar_plane(base="od_out",orientation=orientation, title=title)

def add_water(orientation='y_axes', title="water-density", volume=False):
	global active_water
	active_water = True
	if volume:
		plot_volume(base="wd_out")
	else:
		plot_scalar_plane(base="wd_out",orientation=orientation, title=title)


def add_title():
	global title
	title = mlab.text(0.1, 0.9, "Timestep %d" % time )


def time(t):
	global time
	time = t



def get_screenshot(name="screenshot", antialiased=True, mode='rgb', ext=".png", dpi=800):
	import matplotlib.pylab as plt
	shot = mlab.screenshot(antialiased=antialiased, mode=mode)
	plt.imshow( shot )
	plt.savefig(name+ext, bbox_index='tight', pad_inches=0.1, dpi=dpi)
	plt.close()

def update(continuous=False, sleep=3., step=-1):
	last = [-1, -1]
	while True:
		global time
		if step == -1:
			time = find_data()
		else:
			time = step
		if time == last[0] and time == last[1]:
			print "Found timestep equal to last two times.. aborting."
			break
		clear_figure()

		last[0] = last[-1]
		last[-1] = time
		if active_membrane: add_membrane()
		if active_vel_diff: add_velocity_difference()
		add_title()
		if continuous and step == -1:
			tm.sleep(sleep)
		else:
			break

def animate():
	for t in timesteps[::5]:
		make_frame(t)
		tm.sleep(1)




def get_scalar_field(_filename):
	field = lb3danalysis.readHDF5(_filename)
	shape = field.shape
	xi, yi, zi = np.mgrid[
			1:shape[0]:complex(shape[0]),
			1:shape[1]:complex(shape[1]),
			1:shape[2]:complex(shape[2])]
	return [xi, yi, zi, field]

def add_membrane(alpha=1.0):
	global active_membrane
	active_membrane = True
	membranes = glob.glob('vtk_lagrange/output_p*_t%d.vtk' % time)
	for m in membranes:
		source = mlab.pipeline.open(m)
		particle = mlab.pipeline.surface(source, colormap='Reds', opacity=alpha)

def planes(_filename):
	"""Plots 2D cuts of 3D array data, in the x, y, and z direction. These
	can be moved interactively.
	"""
	xi, yi, zi, field = get_scalar_field(_filename)
	grid = mlab.pipeline.scalar_field(xi, yi, zi, field)
	mlab.pipeline.image_plane_widget(grid,
		plane_orientation='x_axes',
		slice_index = 10,)
	mlab.pipeline.image_plane_widget(grid,
		plane_orientation='y_axes',
		slice_index = 10,)
	mlab.pipeline.image_plane_widget(grid,
		plane_orientation='z_axes',
		slice_index = 10,)

def volume(_filename,_rmin=0.0,_rmax=1.0,_shade=False):
	"""Voxel plot of 3D array data
	"""
	xi, yi, zi, field = get_scalar_field(_filename)
	max = np.max(field)
	min = np.min(field)
	print(max)
	print(min)
	grid = mlab.pipeline.scalar_field(xi, yi, zi, field)
	vol = mlab.pipeline.volume(grid,vmin=min+(max-min)*_rmin,vmax=max*_rmax)
	vol.volume_property.shade = _shade

	#else:
		#mlab.contour3d(xi, yi, zi, field, contours=_contours)

def clear_figure():
	mlab.clf()
	active_membrane = False
	active_vel_diff = False
	active_water = False
	active_oil = False
	active_colour = False

def make_frame(_clear=True):
	if _clear:
		clear_figure()
	#add_membrane()
	#add_title()

def find_data():
	global files
	files = lb3danalysis.getFieldFiles(DATA_PATH)
	#vtks = glob.glob("vtk_lagrange/output_p0_*")
	#timesteps = sorted([ int(re.findall(r'\d+', vtk)[-1]) for vtk in vtks ])
	#print "Last timestep with avaiable data: ", timesteps[-1]
	#return timesteps[-1]

if __name__ == "__main__":
	#global time
	find_data()
	volume(files['elec-phi'][0],0.0,1.0)
	mlab.show()
	#make_frame()
	#mlab.show()
	#if args.time == -1:
	#	time = find_data()
	#else:
#		time = args.time
	#make_frame(time, clear=False)
	#add_title()


#for t in [100, 1000, 10000, 20000]:
#	if os.path.isfile('vtk_lagrange/output_p0_t%d.vtk' % t):

#animation = mpy.VideoClip(make_frame, duration=duration)
#animation.write_videofile("particle.mp4", fps=20)
