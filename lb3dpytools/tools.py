# Module to run, compile, and do other fundamental things for lb3d simulations

import os
import sys # for sys.stdout
import string
import h5py
import numpy as np
import subprocess # used to run the code
import shutil # to copy compiled executable with permissions
import collections # for ordered dictionary, in input file reading/writing

mpirundir = ""
G_exe = "lbe" # name of the executable file, to be modified by used platform
G_elec = False # if compiled with flag -DELEC

# ==========================================================================
# COMPILE
# ==========================================================================
def compile(platform, flags, configDir, sourceDir, out, clean=True, 
		verbose=False, debug=False):
	"""
	Runs the ./configLB3D.sh script according to the platform chosen.
	"""
	global G_exe

	if verbose:
		print("Compiling: "+flags)

	flags_ = string.split(flags)
	if debug:
		flags_.append("-DDEBUG")

	if clean:
		e = subprocess.call(["./configLB3D.sh", "CLEAN"], cwd=configDir, 
				stdout=out, stderr=out)
		e = subprocess.call(["./configLB3D.sh", "CONFIG", platform]+flags_, 
				cwd=configDir, stdout=out, stderr=out)
	e = subprocess.call(["./configLB3D.sh", "MAKE-NODOC"], cwd=configDir, 
			stdout=out, stderr=out)

	if e != 0:
		print("ERROR! Compiling failed!")
	
	return e

# ============================================================================
# RUN
# ============================================================================
def run_command(debug=False):
	if not debug:
		return mpirundir+'mpirun ./'+G_exe+' -f input'
	else:
		valgrindArg = '/apps/prod/HI-ERN/stow/valgrind-3.11.0/bin/valgrind --tool=callgrind'
		return mpirundir+'mpirun '+valgrindArg+' ./'+G_exe+' -f input'
	#return mpirundir+'srun ./'+G_exe+' -f input'

def run(_procs, _runDir, _out, _verbose=False):
	"""
	Runs the code using mpirun, and parameters of number of processors and input file
	"""
	global mpirundir, G_exe

	if _verbose:
		print("Running:")
		sys.stdout.flush()

	#e = subprocess.call([mpirundir+'mpirun -np '+str(_procs)+' ./lbe -f '+
	#_input], cwd=_runDir, stdout=_out, stderr=_out, shell=True)
	e = subprocess.call([mpirundir+'mpirun -np '+str(_procs)+' ./'+G_exe+
		' -f input'], cwd=_runDir, stdout=_out, stderr=_out, shell=True)

	if e != 0:
		print("failed!")
		exit(3)
	else:
		print("done")

# ============================================================================
# RESTORE
# ============================================================================

def restore(name):
	"""
	Restore from a given snapshot
	"""
	global mpirundir, G_exe

	if _verbose:
		print("Restoring sim at snapshot "+name+":")
		sys.stdout.flush()

	e = subprocess.call([mpirundir+'mpirun -np '+str(_procs)+' ./'+G_exe+
		' -f input'], cwd=_runDir, stdout=_out, stderr=_out, shell=True)

	if e != 0:
		print("failed!")
		exit(3)
	else:
		print("done")

# ============================================================================
# DEBUG
# ============================================================================

def debug(_procs, _runDir, _out, _verbose=False, valgrind=False):
	"""
	Debugs the code using valgrind
	"""
	global mpirundir, G_exe

	valgrindArg = ''
	if valgrind:
		valgrindArg = '/apps/prod/HI-ERN/stow/valgrind-3.11.0/bin/valgrind --tool=callgrind'

	if _verbose:
		print("Running (debug mode):")
		sys.stdout.flush()

	#e = subprocess.call([mpirundir+'mpirun -np '+str(_procs)+' ./lbe -f '+
	#_input], cwd=_runDir, stdout=_out, stderr=_out, shell=True)
	#command = mpirundir+'mpirun -np '+str(_procs)+' '+valgrind+' ./'+G_exe+' -f input'
	command = mpirundir+'mpirun -np '+str(_procs)+' '+valgrindArg+' ./'+G_exe+' -f input'
	e = subprocess.call([command], cwd=_runDir, stdout=_out, stderr=_out, shell=True)

	if e != 0:
		print("failed!")
		exit(3)
	else:
		print("done")

# ============================================================================
# ANALYSE
# ============================================================================
def analyse(path, out, verbose=False):
	"""
	Runs the ./analyse.py script that must be in every test directory.
	"""
	if verbose:
		print('Analysing:'),
		sys.stdout.flush()

	if not os.path.isfile(path+'/'+'analyse.py'):
		print('Error: analyse.py was not found')
		exit(1)

	# We call another python script via subprocess; ugly, I know, but we don't
	# want to use 'import' as we really just need to execute it sequentially, once,
	# and care only about a single output number.
	e = subprocess.call('./analyse.py', cwd=path, stdout=out, stderr=out)

	if e != 0:
		print('failed!')
	else:
		print('passed')

# ============================================================================
# BENCHMARK
# ============================================================================
def benchmark(_runDir, _flags, _platform, _procs, _verbose=False):
	if _verbose:
		print('Reading/writing benchmark info')
	
	if not os.path.isfile('log.txt'):
		print('Error: log.txt file not found, we need it to read'
				'benchmark stats')
		exit(1)
	
	fileLog = open('log.txt','r')
	fileBenchmark = open('benchmark.txt','a')
	fileBenchmark.write((datetime.datetime.now()).isoformat()+'\n')
	fileBenchmark.write(_flags+'\n')
	fileBenchmark.write(_platform+'\n')
	fileBenchmark.write(str(_procs)+'\n')
	for line in fileLog:
		if line.find('updates per second') > 0:
			i = line.find('-')
			f = line.find('updates per second')
			n = float(line[i+1:f].strip())
			fileBenchmark.write(str(n)+' ')
	fileBenchmark.write('\n')

# ============================================================================
# INPUT/OUTPUT
# ============================================================================

def outputCheck(_inputDict, _delete=False):
	"""
	Output first check if output directory exists (deletes if _delete, or creates if it doesn't exist)
	"""
	if 'post' in _inputDict:
		if _inputDict['post'] == '.true.':
			outputFolder = _inputDict['folder'][1:-1]
			if not os.path.isdir(outputFolder):
				os.makedirs(outputFolder)
				print 'Warning: output folder not found, so it was created'
			else:
				if _delete:
					shutil.rmtree(outputFolder)
					os.makedirs(outputFolder)

def inputRead(_fileName):
	"""
	Read input to a dictionary, so we can change variable and then print it back.
	"""
	global benchmark

	inputDict = collections.OrderedDict()

	if not os.path.isfile(_fileName):
		print('Error! Input file not found: '+_fileName)
		if benchmark:
			print('Maybe benchmark doesnt exists for this test')
		exit(1)

	inputFile = open(_fileName, 'r')
	for line in inputFile:
		if line[0] == '&':
			inputDict[line] = 'section'
			section = line
		elif line[0] == '/':
			inputDict[section+'/'] = 'break'
		elif line.find('=') > -1:
			var, val = line.split('=')
			inputDict[var.strip()] = val[0:-1].strip()
	return inputDict

def inputWrite(_dictionary, _fileName):
	"""
	Output dictionary to 'file' in the format of an lb3d input file
	"""
	#_file = open(_fileName,'w')
	for key,value in _dictionary.iteritems():
		if key == '&elec_input\n':
			#_file.close()
			_file = open(_fileName+'.elec','w')
		if key == '&md_input\n':
			#_file.close()
			_file = open(_fileName+'.md','w')
		if key == '&fixed_input\n':
			#_file.close()
			_file = open(_fileName,'w')
		if value == 'section':
			_file.write(key)
		elif value == 'break':
			_file.write('/\n\n')
		else:
			_file.write(key+' = '+str(value)+'\n')

	_file.close()

def inputWriteTemporal(_inputFileName, _dictionary):
	"""
	We output to temporal files because we want to retain the original file with
	the original variables (remember we can change variables)
	"""
	inputWrite(_dictionary, 'input-tmp')

	if os.path.isfile(_inputFileName+'.md'):
		shutil.copy2(_inputFileName+'.md','input-tmp.md')

def inputGetDictionaries(_inputFileName, _variables={}):
	"""
	Reads input files to a dictionary, and does all replacements specified in
	the _variable dictionary (all combinations of parameters are run).
	"""
	global G_elec
	
	inputDicts = []
	inputDicts.append(inputRead(_inputFileName))
	# (false comment:) Notice that the other files are put in the same dictionary (as it
	# should have been in the first place) Then we manage to split them
	# when we output
	if os.path.isfile(_inputFileName+'.md'):
		inputDicts.append(inputRead(_inputFileName+'.md'))
		#inputDict.update(inputDictTmp)
	
	if os.path.isfile(_inputFileName+'.elec'):
		inputDicts.append(inputRead(_inputFileName+'.elec'))
		#inputDict.update(inputDictTmp)
	#if G_elec:
	#	inputDictTmp = inputRead(_inputFileName+'.elec')
	#	inputDict.update(inputDictTmp)
	
	
	if not _variables:
		#inputDicts = [inputDict]
		return inputDicts
	else:
		print("VARIABLES IS NOT WORKING")
		exit(0)
		"""
		# Variables of the input file can be modified by the _variable dictionary.
		# Each 'value' can correspond to a list, thus we need to create a mega list
		# with all possible combinations of key/value pairs. This is what we do now:
		varcombs = []
		for key, value in _variables.iteritems():
			if not key in inputDict.keys():
				print("Error! Variable to change is not part of the dictionary of input field names")
				exit(4)
			tmp = []
			if isinstance(value, list):
				for v in value:
					tmp.append([key,v])
			else:
				tmp.append([key,value])
			varcombs.append(tmp)
		varcombs = list(itertools.product(*varcombs))

		# Now create all the possible dictionaries
		for comb in varcombs:
			for v in comb:
				inputDict[v[0]] = v[1]
			inputDicts.append(inputDict)

		return inputDicts
		"""

# ============================================================================
# UTILS
# ============================================================================

class Say:
	WARNING = '\033[93m'
	ERROR = '\033[91m'
	ENDL = '\033[0m'
	obj = None
	def __init__(self, obj):
		self.obj = obj
	#--------------------------------------------------------------------------

	def message(self, str):
		if obj == None:
			print(str)
		else:
			print('p_('+self.obj.name+'): '+str)

	#--------------------------------------------------------------------------

	def message_debug(self, str):
		if obj == None:
			print('(Debug) '+str)
		else:
			print('(Debug) p_('+self.obj.name+'): '+str)

	#--------------------------------------------------------------------------

	def error(self, str):
		if obj == None:
			print(self.ERROR+'Error: '+self.ENDL+str)
		else:
			print('p_('+self.obj.name+'): '+self.ERROR+'Error: '+self.ENDL+str)

	#--------------------------------------------------------------------------

	def warning(self, str):
		if obj == None:
			print(self.WARNING+'Warning! '+self.ENDL+str)
		else:
			print(self.WARNING+'p_('+self.obj.name+'): Warning! '+self.ENDL+str)

	#--------------------------------------------------------------------------

def setMPI(platform):
	global mpirundir

	if platform == 'SUN':
		mpirundir = '/usr/lib/openmpi/1.8.4/common/bin/'
	elif platform == 'SUN-INTEL' or platform == 'SUN-INTEL-DEBUG':
		mpirundir ='/usr/lib/openmpi/1.8.8/common/bin/'
	else:
		mpirundir = ''
	# We search that mpirun actually exists
	mpirunExists = False
	for dir in os.environ["PATH"].split(":"):
		if (os.path.isfile(dir+"/mpirun")):
			mpirunExists = True
	if not mpirunExists:
		if not os.path.isfile(mpirundir+"/mpirun"):
			print("Error: mpirun not found")
			return 1
	return 0

def readHDF5(_file):
	"""
	Given a hdf5 filename, return a numpy array with the data
	"""
	if os.path.isfile(_file):
		if _file.find(".h5") > 0:
			dat = []
			h5f = h5py.File(_file, "r")
			dat = h5f["OutArray"][:] # TODO: Is this always the case?
			h5f.close()
			return np.swapaxes(dat, 0, 2)
		else:
			print("(readHDF5) Error: file extension does not match hdf5")
			return -1
	else:
		print("(readHDF5) Error: file not found")
		return -1
