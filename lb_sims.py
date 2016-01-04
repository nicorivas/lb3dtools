#!/usr/bin/env python

# This package defines the Project and Simulation classes, very useful to deal
# with simulations and collection of simulations (project) on python.

import sys
import os
import datetime # to print time stamps in history files
import shutil # to copy files with permissions
import subprocess # to execute batch file when queue is called
import glob

import lb_tools # see module

verbose = False

# Global constants
DATA_DIR = "/data/home/nicorivas/Data"
#DATA_DIR = "/data/home/nicorivas/Code/lb3d"
#SOURCE_DIR = "/data/home/nicorivas/Code/lb3d/src"
SOURCE_DIR = "/Users/local_admin/Code/lb3d/src"
#CONFIG_DIR = "/data/home/nicorivas/Code/lb3d"
CONFIG_DIR = "/Users/local_admin/Code/lb3d"

class Project:
    name = "" # name of the simulation, which is also directory name
    directory = "" # full path
    projectFilename = "" # project file, almost only a readme
    historyFilename = "" # history file name
    sc = 0
    sims = []
    loaded = False

    def __init__(self):
        """
        Constructor does nothing, to enforce that the user does either new or load
        to have the project.
        """
        if verbose:
            print("Project constructor "+str(name))

    def new(self, name):
        """
        Given a name, it creates a project
        """
        self.name = name
        self.directory = DATA_DIR+"/"+name
        if os.path.exists(self.directory):
            print("Project already exists! Use .load() to load a project")
            return 1
        self.projectFilename = self.directory+'/project'
        self.historyFilename = self.directory+'/history'

        # CREATE DIRECTORY AND FILES

        return self

    def load(self, name):
        self.name = name
        self.directory = DATA_DIR+"/"+name
        if not os.path.exists(self.directory):
            print("Project does not exist! Create it?")
            return 1
        self.projectFilename = self.directory+'/project'
        if not os.path.isfile(self.projectFilename):
            print("Project file not found. Are you sure is this a project directory? Aborting")
            return 1
        self.historyFilename = self.directory+'/history'
        if not os.path.isfile(self.historyFilename):
            print("History file not found. Are you sure is this a project directory? Aborting")
            return 1
        if verbose:
            print("Loaded project in "+self.directory)

        self.loaded = True

        # Look for simulations
        directories = glob.glob(self.directory+"/*")
        for dir in directories:
            r = self.addSimulation(dir)
            if r != 0:
                print("WARNING: Directory "+dir+" is not a simulation!")

        return self

    def writeToFile(self, string):
        file = open(self.projectFilename,'w')
        file.write(string)
        file.close()

    def writeToHistory(self, string):
        if (os.path.exists(self.historyFilename)):
            file = open(self.historyFilename,'a')
        else:
            file = open(self.historyFilename,'w')
        time = datetime.datetime.now()
        file.write(str(time)+': '+string+'\n')
        file.close()

    def addSimulation(self, name):

        path = self.directory+"/"+name

        if not os.path.exists(path):
            print("ERROR: Simulation path not found")
            return 1
        else:
            sims.sims.append(Simulation.new(name, self))
            self.sc += 1
            return 0

    def __str__(self):
        return  "Project instance:\n"\
                "\tname: "+self.name+"\n"\
                "\tdirectory: "+self.directory+"\n"\
                "\tprojectFilename: "+self.projectFilename+"\n"\
                "\thistoryFilename: "+self.historyFilename+"\n"

#==============================================================================

class Simulation:

    CLASS_NAME = "Simulation"
    debug = True

    def __init__(self, project, name):
        """It is not clear yet how much the constructor should do, I will keep
           tweaking it as performance becomes an issue. That's why we call other
           functions, to have more flexibility.
             @project: Project object
             @name: name of the simulation and directory (same thing always)
        """
        # Instance variables
        self.project = None
        self.name = ""
        self.directory = ""
        self.template = ""
        self.platform = ""
        self.flags = ""
        self.procs = 4
        self.inputFile = ""
        # key's are the field name (i.e. 'eps'), and it contains a list with all
        # the filenames of the respective field (for every time, and maybe also
        # run and name)
        self.fieldFiles = {}
        self.inputDicts = {}
        self.outputDir = ""
        self.compiled = False
        self.verbose = True

        self.name = name
        self.directory = project.directory+"/"+name
        self.project = project

        """
        if os.path.exists(self.directory):
            print("Simulation already exists! So loading it")
            self.load()
        else:
            print("Simulation doesn't exist! So creating it")
            os.makedirs(self.directory)
            project.writeToHistory(self.name+' created')
        """
#______________________________________________________________________________

    def load(self):
        """Determine variables from the directory, specially the dictionaries
           from the input files, and the filenames of the output data
        """
        if self.debug:
            print(self.CLASS_NAME+"::load()")

        if not os.path.exists(self.directory):
            print("Directory not found!: "+self.directory)
            return -1

        self.inputFile = self.directory+"/input"
        if os.path.isfile(self.inputFile):
            self.inputDicts = lb_tools.inputGetDictionaries(self.inputFile)
            self.outputDir = self.inputDicts[0]['folder'][1:-1]
            if (os.path.exists(self.directory+'/'+self.outputDir)):
                self.loadFieldFiles()
#______________________________________________________________________________

    def loadFieldFiles(self, prefix="*"):
        """Load field filenames to dictionary. For format of the dictionary
           see its definition, up.
             @prefix: for selecting by wildcards
        """
        if self.debug:
            print(self.CLASS_NAME+"::loadFieldFiles()")

        # Not neccesarely present, just all possible output prefixes from lb3d
        field_names_ = ['flooil','arr','od','rock_state','vel','velx','vely',
                        'velz','elec-phi','elec-rho_p','elec-rho_m',
                        'elec-init','elec-eq_postNoE','elec-eq_postPoisson',
                        'elec-eq_init','elec-eq_final']
        for field_name_ in field_names_:
            path_ = self.directory+'/'+self.outputDir+'/'+field_name_+'_'+prefix+'.h5'
            files_ = glob.glob(path_)
            if len(files_) > 0:
                self.fieldFiles[field_name_] = files_
#______________________________________________________________________________

    def loadTemplate(self, path):
        """A template is an unrun simulation with a flags file that also
           contains the preprocessor compilation flags
        """
        if self.debug:
            print(self.CLASS_NAME+"::loadTemplate()")

        if not os.path.exists(path):
            print("Template not found!")
            exit(1)

        self.template = path
        self.inputFile = path+"/input"
        self.inputDicts = lb_tools.inputGetDictionaries(self.inputFile)
        # read preprocessor files
        f = open(self.template+'/flags','r')
        self.flags = f.read().strip()
#______________________________________________________________________________

    def getField(self, fieldname, t=0, format='hdf5'):
        """Given a Simulation object, and name, returns the corresponding
           field as numpy array
             @fieldname: name of the field, as prefix in lb3d output files
             @t: index from total fields for the particular file to get
             @format: for different file types
        """
        field_ = []
        if format == 'hdf5':
            if fieldname in self.fieldFiles.keys():
                field_ = lb_tools.readHDF5(self.fieldFiles[fieldname][t])
            else:
                print("WARNING! Field not found")
                if self.debug:
                    print(fieldname)
                    print(self.fieldFiles.keys())
        else:
            print("ERROR! Format not supported")
        return field_
#______________________________________________________________________________

    def copyBinary(self, filename='lbe'):
        # We use .copy2 because we also want to copy permissions
        # Copy executable
        print(SOURCE_DIR)
        print(self.directory)
        shutil.copy2(SOURCE_DIR+'/lbe', self.directory)
        os.rename(self.directory+'/'+'lbe', self.directory+'/'+filename)

#______________________________________________________________________________

    def commit(self):
        if self.debug:
            print(self.CLASS_NAME+"::commit()")

        if not os.path.exists(self.directory):
            print("Simulation doesn't exist! So creating it")
            os.makedirs(self.directory)
            self.project.writeToHistory(self.name+' created')

        # We use .copy2 because we also want to copy permissions
        # Copy executable
        print(self.directory)
        shutil.copy2(SOURCE_DIR+'/lbe', self.directory+'/')

        # Update configuration files
        lb_tools.inputWrite(self.inputDicts[0], self.directory+'/input')

        #shutil.copy2(self.inputFile, self.directory)
        if os.path.isfile(self.inputFile+'.md'):
            lb_tools.inputWrite(self.inputDicts[1], self.directory+'/input')
            #shutil.copy2(self.inputFile+'.md', self.directory)
            mdfile = self.template+"/"+self.inputDicts[1]['init_file'][1:-1]
            shutil.copy2(mdfile, self.directory)

        if os.path.isfile(self.inputFile+'.elec'):
            lb_tools.inputWrite(self.inputDicts[2], self.directory+'/input')
            #shutil.copy2(self.inputFile+'.elec', self.directory)

        # Create output directory
        outputdir = self.inputDicts[0]['folder'][1:-1]
        if not os.path.exists(self.directory+"/"+outputdir):
            os.makedirs(self.directory+"/"+outputdir)
#______________________________________________________________________________

    def set(self, dicn, key, value):
        self.inputDicts[dicn][key] = value
#______________________________________________________________________________

    def get(self, dicn, key):
        if key in self.inputDicts[dicn].keys():
           return self.inputDicts[dicn][key]
        else:
            print("ERROR: Key not found")
            return -1
#______________________________________________________________________________

    def queue(self):
        """Instead of running, submits jobs to queue. This works by creating
           a bash script that sbatch can understand, with parameters relevant
           for the run
        """
        return_code_ = 0 # return from calling program
        bath_filename_ = "" # name of the bash script file to be run

        # Notice here that we set the directory to the simulation
        os.chdir(self.directory)

        lb_tools.setMPI(self.platform)

        batch_filename_ = self.project.name+'-'+self.name

        # Write sbatch queue
        f = open(batch_filename_, 'w')
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -n '+str(self.procs)+'\n')
        f.write('#SBATCH -o debug-%N-%j.out\n')
        f.write('#SBATCH -e debug-%N-%j.err\n')
        f.write('#SBATCH -J '+self.name+'\n')
        f.write('#SBATCH --get-user-env\n')
        f.write('#SBATCH --time=24:00:00\n')
        #f.write('srun ./lbe -f input')
        f.write(lb_tools.run_command())
        f.close()

        # And run it
        return_code_ = subprocess.call("sbatch "+batch_filename_, shell=True)
        if return_code_ != 0:
            print("Something went wrong when calling sbatch. I quit!")
            exit(1)
#______________________________________________________________________________

    def compile(self, clean=False):
        if not self.compiled:
            lb_tools.compile(self.platform, self.flags, CONFIG_DIR, SOURCE_DIR,
                sys.stdout, clean=True, verbose=False)
#______________________________________________________________________________

    def debug(self):
        lb_tools.setMPI(self.platform)
        lb_tools.debug(self.procs, self.directory, sys.stdout, False)
#______________________________________________________________________________

    def run(self):
        lb_tools.setMPI(self.platform)
        lb_tools.run(self.procs, self.directory, sys.stdout, False)
#______________________________________________________________________________

    def analyse(self):
        lb_tools.analyse(self.directory, sys.stdout, False)

#______________________________________________________________________________

    def __str__(self):
        """Called by print(instance)
        """
        return  "Simulation instance:\n"\
                "\tname: "+self.name+"\n"\
                "\tdirectory: "+self.directory+"\n"\
                "\tflags: "+self.flags+"\n"\
                "\tinputFile: "+self.inputFile+"\n"\
                "\tinputDicts: "+str(self.inputDicts)

#==============================================================================
