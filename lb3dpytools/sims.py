#!/usr/bin/env python

# This package defines the Project and Simulation classes, very useful to deal
# with simulations and collection of simulations (projects) on Python.

import sys
import os
import datetime   # to print time stamps in history files
import shutil     # to copy files with permissions
import subprocess # to execute batch file when queue is called
import glob
import numpy as np
# Local to module:
import tools # see module

verbose = False

# Global constants
DATA_DIR = "/data/nicorivas/"
#DATA_DIR = "/homec/jiek11/jiek1101/Data"

SOURCE_DIR = "/data/home/nicorivas/Code/lb3d/src"
#SOURCE_DIR = "/homec/jiek11/jiek1101/Code/lb3d/src"
#SOURCE_DIR = "/Users/local_admin/Code/lb3d/src"

CONFIG_DIR = "/data/home/nicorivas/Code/lb3d"
#CONFIG_DIR = "/homec/jiek11/jiek1101/Code/lb3d"
#CONFIG_DIR = "/Users/local_admin/Code/lb3d"

class Project:
    """Class to hold collections of simulation, defined by a directory.
    The class project is useful to manage directories with many simualtions,
    which are here refered to as projects. It manages a history file for
    every simulation that is run, modified, deleted, or others. It also has
    an array with all simulations in the directory, which can be useful
    when analysing data.
    """

    CLASS_NAME = 'Project'
    DEBUG = False

    #--------------------------------------------------------------------------

    def __init__(self):
        """
        Constructor does nothing, to enforce that the user does either new or
        load to have a project, otherwise dangereous things could happen.
        """
        self.name = "" # name of the simulation, which is also directory name
        self.directory = "" # full path
        self.projectFilename = "" # project file, almost only a readme
        self.historyFilename = "" # history file name
        self.sc = 0 # number of simulations
        self.sims = [] # list of simulation objects
        self.loaded = False # if it has been loaded
        self.say = tools.Say(self)

    #--------------------------------------------------------------------------

    def new(self, name):
        """
        Given a name, it creates a project, which involves creating the
        directory and the 'history' and 'project' files
        """
        self.name = name
        self.directory = DATA_DIR+"/"+name
        if os.path.exists(self.directory):
            self.error("Project already exists! Use .load() to load a project")
            return 1
        self.projectFilename = self.directory+'/project'
        self.historyFilename = self.directory+'/history'

        # Create directory
        try:
            os.makedirs(self.directory)
        except OSError as e:
            self.say.error('Could create directory when creating project: {}'.format(e))
            exit(0)

        # Log and project files
        self.writeToFile("Project file\n")
        self.writeToHistory("Project created\n")

        return self

    #--------------------------------------------------------------------------

    def load(self, name):
        """Loads a project by name.
        Given a name, we know the directory to look for, and we add all
        simulations to the sims array as Simulation objects.
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::load(name="+name+")")

        self.name = name
        self.directory = DATA_DIR+"/"+name
        if not os.path.exists(self.directory):
            self.say.error("Project does not exist ("+self.directory+")!")
            return 1
        self.projectFilename = self.directory+'/project'
        if not os.path.isfile(self.projectFilename):
            self.say.error("Project file not found."
                    " Are you sure is this a project directory? Aborting")
            return 1
        self.historyFilename = self.directory+'/history'
        if not os.path.isfile(self.historyFilename):
            self.say.error("History file not found."
                    " Are you sure is this a project directory? Aborting")
            return 1

        if verbose:
            self.say.message("Loaded ('"+self.directory+"')")

        self.loaded = True

        return self

    #--------------------------------------------------------------------------

    def loadSimulations(self):
        """Loads all simulations in the project's directory.
        Load simulations take the currenct directory and adds all the
        directories that are found to be a simulation. We complain for
        non-project files in the directory. TODO: Should we?
        """
        directories = glob.glob(self.directory+"/*")
        for dir in directories:
            if os.path.isdir(dir):
                # Load simulations
                r = self.addSimulation(dir)
                if r == 0:
                    r;
                    #self.message("Simulation loaded: '"+dir+"'")
            else:
                # Found a file, if it's not project or history files, complain
                filename = os.path.split(dir)[1]
                if not filename == 'project' and not filename == 'history':
                    if self.DEBUG:
                        self.say.warning("Extra file found on project directory ("+dir+")")
        return 0

    #--------------------------------------------------------------------------

    def addSimulation(self, path):
        """Add a simulation to the project, that is, to the array 'sims'.
        Make sure it exists first.
        Notice that simulations are also loaded (.load is called in Simulation)
        """
        if self.DEBUG:
            self.message_debug(self.CLASS_NAME+"::addSimulation(path="+path+")")

        if not os.path.exists(path):
            self.error("While adding simulation: path not found ("+path+")")
            return 1
        elif not os.path.isdir(path):
            self.error("While adding simulation: trying to load a file ("+path+")")
            return 1
        elif not os.path.isfile(path+'/input'):
            self.error("Directory '"+path+"' is not a simulation")
            return 1
        else:
            name_ = os.path.split(path)[1]
            self.sims.append(Simulation())
            self.sims[self.sc].load(path)
            self.sims[self.sc].setProject(self)
            self.sims[self.sc].setName(name_)
            self.sc += 1
        return 0

    #--------------------------------------------------------------------------

    def selectSimulations(self, params_col):
        """
        Given a list parameters sets, get all simulations that fulfill all
        conditions. Parameter sets are of the form:
            [...[dict_number, variable_name, variable_value]...]
        """
        sims_all = self.sims
        sims_col = []
        for params in params_col:
            sims = sims_all
            for param in params:
                sims = [sim for sim in sims if sim.get(param[0],param[1]) == str(param[2])]
            sims_col += sims
        return sims_col

    #--------------------------------------------------------------------------

    def writeToFile(self, string):
        """Write the given string to the project file.
        Quite useless now but eventually useful.
        """
        file = open(self.projectFilename,'w')
        file.write(string)
        file.close()

    #--------------------------------------------------------------------------

    def writeToHistory(self, string, sim=""):
        """
        Write string to the project's history file, with an added timestamp
        Optional 'sim' argument can be given to identify which simulation
        is writing here.
        """
        if (os.path.exists(self.historyFilename)):
            file = open(self.historyFilename,'a')
        else:
            file = open(self.historyFilename,'w')
        time = datetime.datetime.now()
        file.write(str(time)+': ('+sim+') '+string+'\n')
        file.close()

    def __str__(self):
        return  "Project instance:\n"\
                "\tname: "+self.name+"\n"\
                "\tdirectory: "+self.directory+"\n"\
                "\tprojectFilename: "+self.projectFilename+"\n"\
                "\thistoryFilename: "+self.historyFilename+"\n"

#==============================================================================

class Simulation:
    """
    Simulation objects are associated with a lb3d simulation directory.
    They hold the input files in a dictionary, the compiler flags as a string,
    and other configuration options. Very useful to run, get properties,
    get output data and analyse it.
    """

    CLASS_NAME = "Simulation"
    DEBUG = False

    #--------------------------------------------------------------------------

    def __init__(self):
        """
        Constructor just initializes the used variables and does nothing else
        """
        # Instance variables
        self.project = None
        self.name = ""
        self.directory = ""
        self.template = ""
        self.platform = ""
        self.flags = ""
        self.inputFile = ""
        # key's are the field name (i.e. 'eps'), and it contains a list with all
        # the filenames of the respective field (for every time, and maybe also
        # run and name)
        self.fieldFiles = {}
            # Dictionary from output field prefix name to list of filenames
        self.fieldFilesN = {}
            # Dictionary from output field prefix name to number of files
        self.outputLoaded = False
        self.inputDicts = {}
        self.outputDir = ""
        self.compiled = False
        self.verbose = True
        self.id = 0 # Id as printed in output files, for restoring sims
        self.restoreFiles = {}
        self.say = tools.Say(self)

    #--------------------------------------------------------------------------

    def new(self, project, name):
        """
        New just sets the name and project variables (as well as the path),
        but does not create directories, because it makes things easier
        """
        self.setProject(project)
        r = self.setName(name)
        return r

    #--------------------------------------------------------------------------

    def setProject(self, project):
        """ In case we loaded the single simulation, assign the project
        """
        self.project = project

    #--------------------------------------------------------------------------

    def setName(self, name):
        """ Setter that also sets the directory name (which is always the same)
        """
        self.name = name

        if self.project is not None:
            self.directory = self.project.directory+"/"+self.name
        else:
            self.say.error("Setting name, but project is not set!")
            return 1

        return 0

    #--------------------------------------------------------------------------

    def load(self, path, input='input'):
        """
        Determine variables from the directory, specially the dictionaries
        from the input files, and the filenames of the output data
        """
        if self.DEBUG:
            self.message_debug(self.CLASS_NAME+"::load(path="+path+")")

        if not os.path.isdir(path):
            self.say.error("While loading: directory does not exist! ("+path+")")
            return 1
        else:
            self.directory = path
            self.name = os.path.split(self.directory)[1]

        if self.loadInput(input):
            return 1

        return 0

    #--------------------------------------------------------------------------

    def loadInput(self, input='input'):
        """
        Loads input files, using 'tools'. It is not neccesary to do this for
        all input files (i.e. elec), as it checks if they already exist
        """
        self.inputFile = self.directory+"/"+input
        if os.path.isfile(self.inputFile):
            self.inputDicts = tools.inputGetDictionaries(self.inputFile)
            return 0
        else:
            self.say.error("input file not found")
            return 1

    #--------------------------------------------------------------------------

    def loadOutput(self, prefix="*"):
        """
        Load into the simulation class output data.
        First check if  output directory exists.
        """
        if not self.outputLoaded:
            if 'folder' in self.inputDicts[0].keys():
                self.outputDir = self.inputDicts[0]['folder'][1:-1]
            else:
                self.outputDir = 'Production'

            if (os.path.exists(self.directory+'/'+self.outputDir)):
                self.loadFieldFiles(prefix)
                self.outputLoaded = True
                return 0
            else:
                return 1
        else:
            if self.verbose:
                self.say.warning('Tried to load output when already loaded')

    #--------------------------------------------------------------------------

    def loadFieldFiles(self, prefix="*"):
        """
        Load field filenames to dictionary. For format of the dictionary
        see its definition, up.
            @prefix: for selecting by wildcards
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::loadFieldFiles()")

        # Not neccesarely present, just all possible output prefixes from lb3d
        field_names_ = ['flooil','arr',
                        'od','wd',
                        'rock_state',
                        'vel','velx','vely','velz',
                        'velx_od','vely_od','velz_od',
                        'force',
                        'Fx','Fy','Fz',
                        'elec-rho_p',
                        'elec-rho_m',
                        'elec-phi',
                        'elec-eps',
                        'elec-Ex','elec-Ey','elec-Ez',
                        'elec-init-post_eps',
                        'elec-init-post_rho_p',
                        'elec-init-post_rho_m',
                        'elec-init-post_phi',
                        'elec-init-post_Ex',
                        'elec-init-post_Ey',
                        'elec-init-post_Ez',
                        'elec-eq_postPoisson_eps',
                        'elec-eq_postPoisson_rho_p',
                        'elec-eq_postPoisson_rho_m',
                        'elec-eq_postPoisson_phi',
                        'elec-eq_postPoisson_Ex',
                        'elec-eq_postPoisson_Ey',
                        'elec-eq_postPoisson_Ez',
                        'elec-eq_postNoE_eps',
                        'elec-eq_postNoE_rho_p',
                        'elec-eq_postNoE_rho_m',
                        'elec-eq_postNoE_phi',
                        'elec-eq_postNoE_Ex',
                        'elec-eq_postNoE_Ey',
                        'elec-eq_postNoE_Ez',
                        'elec-eq_init_eps',
                        'elec-eq_init_rho_p',
                        'elec-eq_init_rho_m',
                        'elec-eq_init_phi',
                        'elec-eq_init_Ex',
                        'elec-eq_init_Ey',
                        'elec-eq_init_Ez',
                        'elec-eq_final_eps',
                        'elec-eq_final_rho_p',
                        'elec-eq_final_rho_m',
                        'elec-eq_final_phi',
                        'elec-eq_final_Ex',
                        'elec-eq_final_Ey',
                        'elec-eq_final_Ez']

        for field_name_ in field_names_:
            path_ = self.directory+'/'+self.outputDir+'/'+field_name_+'_'+prefix+'.h5'
            files_ = glob.glob(path_)
            files_.sort()
            if len(files_) > 0:
                self.fieldFiles[field_name_] = files_
                self.fieldFilesN[field_name_] = len(files_)

    #--------------------------------------------------------------------------

    def loadTemplate(self, path):
        """
        A template is an simulation directory with a flags file that contain
        the preprocessor compilation flags. Useful to not define the whole
        input file every time we want to run a simulation.
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::loadTemplate()")

        if not os.path.exists(path):
            self.say.error("Template not found! ("+path+")")
            exit(1)

        self.template = path
        self.inputFile = path+"/input"
        self.inputDicts = tools.inputGetDictionaries(self.inputFile)

        # read preprocessor directives
        if self.flags == "":
            f = open(self.template+'/flags','r')
            self.flags = f.read().strip()
        else:
            self.warning("Flags were not read from file; already set")

        self.message("Template loaded: '"+path+"'")

    #--------------------------------------------------------------------------

    def getField(self, fieldname, t=0, format='hdf5'):
        """
        Given a Simulation object, and name, returns the corresponding
        field as numpy array
          @fieldname: name of the field, as prefix in lb3d output files
          @t: index from total fields for the particular file to get
          @format: for different file types
        """
        if self.DEBUG:
            self.message_debug(self.CLASS_NAME+"::getField(fieldname="+fieldname+")")

        if self.fieldFiles.keys() == []:
            self.say.error("Looks like you haven't loaded the output, use loadOutput")
            return -1

        field_ = []
        if format == 'hdf5':
            if fieldname in self.fieldFiles.keys():
                if t < len(self.fieldFiles[fieldname]):
                    fn_ = self.fieldFiles[fieldname][t]
                else:
                    self.say.error("Time is out of joint (for '{}' at {})".format(fieldname, t))
                    return -1
                field_ = tools.readHDF5(fn_)
                if self.DEBUG:
                    self.say.message_debug(fn_)
            else:
                self.say.warning("Field not found '"+fieldname+"'")
                return -1
        else:
            self.say.error("Format not supported")
            return -1
        return field_

    #--------------------------------------------------------------------------

    def copyBinary(self, filename='lbe'):
        """
        Copy the binary from the source dir, to the current simulation dir
        """
        # We use .copy2 because we also want to copy permissions
        # Copy executable
        shutil.copy2(SOURCE_DIR+'/lbe', self.directory)
        os.rename(self.directory+'/'+'lbe', self.directory+'/'+filename)
        self.project.writeToHistory("Executable was overwritten",sim=self.name)

    #--------------------------------------------------------------------------

    def commit(self,deleteOutput=True,restore=False):
        """
        Commit creates the directories, outputs the input files, and copies
        the executable file if it was compiled
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::commit()")

        if not os.path.exists(self.directory):
            if restore:
                self.say.error("Set to restore, but simulation does not exist!")
                exit(0)
            else:
                self.say.message("Simulation directory doesn't exist! So creating it")
                os.makedirs(self.directory)
                self.project.writeToHistory(self.name+' created')

        if restore and deleteOutput:
            self.say.warning("You really don't want to delete output when"
                    "restoring! So we changed it")
            deleteOutput = False

        # We use .copy2 because we also want to copy permissions
        # Copy executable
        if self.compiled:
            self.copyBinary()

        # Update configuration files
        tools.inputWrite(self.inputDicts[0], self.directory+'/input')
        c_ = 1
        # Md config files
        if os.path.isfile(self.inputFile+'.md'):
            tools.inputWrite(self.inputDicts[c_], self.directory+'/input')
            #shutil.copy2(self.inputFile+'.md', self.directory)
            if self.template is not '':
                mdfile = self.template+"/"+self.inputDicts[c_]['init_file'][1:-1]
                shutil.copy2(mdfile, self.directory)
            c_ += 1
        # Elec config files
        if os.path.isfile(self.inputFile+'.elec'):
            tools.inputWrite(self.inputDicts[c_], self.directory+'/input')
            #shutil.copy2(self.inputFile+'.elec', self.directory)
            c_ += 1
        self.project.writeToHistory("Configuration files were written",sim=self.name)

        self.project.writeToHistory("Simulation commited",sim=self.name)

        # Create output directory
        # If it exists, do we delete the files?
        outputdir = self.inputDicts[0]['folder'][1:-1]
        fd = self.directory+"/"+outputdir
        if os.path.exists(fd):
            if deleteOutput:
                if len(outputdir) > 1:
                    shutil.rmtree(fd)
                    self.project.writeToHistory("Output dir was removed!",sim=self.name)
                    os.makedirs(fd)
            else:
                if not restore:
                    self.say.warning("Output dir existed, we are adding files there")
        else:
            os.makedirs(fd)

    #--------------------------------------------------------------------------

    def set(self, dicn, key, value):
        self.inputDicts[dicn][key] = value

    #--------------------------------------------------------------------------

    def get(self, dicn, key):
        """
        Get a variable from the input files.
        Select a dictionary with 'dicn', from input, input.elec, input.md, etc.
        """
        if key in self.inputDicts[dicn].keys():
            return self.inputDicts[dicn][key]
        else:
            self.say.error("Property key '"+key+"' not found")
            return -1

    #--------------------------------------------------------------------------

    def queue(self,
            procs=4,
            node='',
            nodes=1,
            tasks_per_node=24,
            time='24:00:00',
            exclusive=False,
            debug=False):
        """
        Instead of running, submits jobs to queue (SLURM). Creates a bash
        script with SBATCH commands relevant for the run.
        For SBATCH doc see https://computing.llnl.gov/linux/slurm/sbatch.html
            @node: name of the node to be used in the cluster. This can be a
                   list of names separated by commas.

        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::queue()")

        return_code_ = 0        # return from calling program
        batch_filename_ = ""    # name of the bash script file to be run

        # !! Notice here that we set the directory to the simulation
        os.chdir(self.directory)

        foundmpi = tools.setMPI(self.platform)

        batch_filename_ = self.project.name+'-'+self.name

        # Write sbatch queue
        f = open(batch_filename_, 'w')
        f.write('#!/bin/bash\n')
        #f.write('#SBATCH --cpu_bind=rank\n') # needs plug-in
        if exclusive:
            f.write('#SBATCH --exclusive\n')
            # don't share nodes with other procs
        f.write('#SBATCH --nodes='+str(nodes)+'\n') # number of nodes to use
        f.write('#SBATCH --ntasks='+str(procs)+'\n')
            # number of total tasks, so 'x' if running ./mpirun -n x.
        #f.write('#SBATCH --ntasks-per-node='+str(tasks_per_node)+'\n')
            # tasks per node; if this is set low enough the no hyperthreading
            # takes place. this should work, but simulations sometime crash
        #f.write('#SBATCH --cpus-per-task=1\n')
            # this should work, but simulations sometime crash before starting
            # if set. i couldn't find the origin of the crash.
        f.write('#SBATCH --output debug-%N-%j.out\n') # stdout redirect
        f.write('#SBATCH --error debug-%N-%j.err\n') # stderr redirect
        f.write('#SBATCH --job-name '+self.name+'\n') # name for queue (8c max)
        #f.write('#SBATCH --get-user-env\n') # use env variables if set
        f.write('#SBATCH --time='+time+'\n') # max time: a day
        if node is not '':
            f.write('#SBATCH --nodelist='+node+'\n')
        if foundmpi==0:
            f.write(tools.run_command(debug=debug))
        else:
            f.write('srun ./lbe -f input')
        f.close()

        self.project.writeToHistory("Batch file created",sim=self.name)

        self.message("Adding to queue:")

        self.project.writeToHistory("Running batch file...",sim=self.name)

        # And run it
        return_code_ = subprocess.call("sbatch "+batch_filename_, shell=True)
        if return_code_ != 0:
            self.say.error("Something went wrong when calling sbatch.")
            return return_code_

        self.project.writeToHistory("Batch file finished!",sim=self.name)

        return return_code_

    #--------------------------------------------------------------------------

    def compile(self, clean=False, debug=False, out=None):
        """
        Call the 'tools' 'compile' routine with proper arguments. See doc there.
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::compile()")

        self.message("Compiling...")
        print("-"*80)

        if out == None:
            out = open(os.devnull,'w')

        r = 1
        if not self.compiled:
            r = tools.compile(self.platform, self.flags, CONFIG_DIR, SOURCE_DIR,
                out, clean=clean, debug=debug, verbose=False)
            if r != 0:
                self.project.writeToHistory("Error: Compilation failed",sim=self.name)
            else:
                print("-"*80)
                self.project.writeToHistory("Compiled source with flags "+str(self.flags),sim=self.name)
                self.compiled = True
        return r

    #--------------------------------------------------------------------------

    def branch(self, target, time='0'):
        """ Branch takes simulation 'target' and copies the checkpoint files to current simulation
        """
        if len(self.restoreFiles) == 0:
            self.say.error('Restore files not loaded or not existing')
        if str(time) not in self.restoreFiles.keys():
            self.say.error('Time {} not found in restore files'.format(time))
        for file in self.restoreFiles[str(time)]:
            shutil.copy2(file, target+'/output')

    #--------------------------------------------------------------------------

    def loadRestore(self):
        """
        Check if there are restore files, and get their ID
        """
        files = glob.glob(self.directory+'/output/cp_*.h5')
        files = sorted(files)
        if len(files) == 0:
            self.say.error('Restore files not found')
            exit(0)

        # get id
        file = files[0]
        self.id = int(file[file.find('_t')+11:file.find('.h5')])

        # get times and set dictionary
        files = glob.glob(self.directory+'/output/cp_*.h5')
        for file in files:
            # str(int(x)) to remove the trailing zeros
            time = str(int(file[file.find('_t')+2:file.find('.h5')-11]))
            if time not in self.restoreFiles.keys():
                self.restoreFiles[time] = []
            self.restoreFiles[time].append(file)

    #--------------------------------------------------------------------------

    def restore(self, name):
        """ Just a reference
        """
        tools.restore(name)

    #--------------------------------------------------------------------------

    def debug(self, procs=4, valgrind=False):
        """ Call the 'tools' 'debug' routine with proper arguments. See doc there.
        """
        tools.setMPI(self.platform)
        tools.debug(procs, self.directory, sys.stdout,
                _verbose=False, valgrind=valgrind)

    #--------------------------------------------------------------------------

    def run(self, procs=4, out=sys.stdout):
        """ Call the 'tools' 'run' routine with proper arguments. See doc there.
        """
        if self.DEBUG:
            self.say.message_debug(self.CLASS_NAME+"::run()")

        if out == None:
            out = open(os.devnull,'w')

        self.project.writeToHistory("Run started",sim=self.name)

        tools.setMPI(self.platform)
        tools.run(procs, self.directory, out, False)

        self.project.writeToHistory("Run finished!",sim=self.name)

    #--------------------------------------------------------------------------

    def analyse(self):
        """ Call the 'tools' 'analyse' routine with proper arguments. See doc there.
        """
        tools.analyse(self.directory, sys.stdout, False)

    #--------------------------------------------------------------------------

    def getMDData(self):
        """ Read MD snapshot files.
        Returns an array with the structure a[time][particle] = [x,y,z,...]
        """
        if not self.outputLoaded:
            self.say.error("Output doesn't seem to be loaded")
            return -1

        filenames = glob.glob(self.directory+'/'+self.outputDir+"/md-cfg_output*")
        filenames = sorted(filenames)

        # determine the number of particles (we assume it stays constant!)
        pn = 0
        f = open(filenames[0],'r')
        for l in f:
            pn += 1
        f.close()

        md_data = np.zeros([len(filenames),pn,7])
        ti = 0
        for filename in filenames:
            pi = 0
            f = open(filename, 'r')
            for line in f:
                ls = line.split()
                ls = [float(s) for s in ls]
                md_data[ti][pi] = ls
                pi += 1
            ti += 1
        return md_data

    #--------------------------------------------------------------------------

    def __str__(self):
        """ Called by print(instance)
        """
        return  "Simulation instance:\n"\
                "\tname: "+self.name+"\n"\
                "\tdirectory: "+self.directory+"\n"\
                "\tflags: "+self.flags+"\n"\
                "\tinputFile: "+self.inputFile+"\n"\
                "\tinputDicts: "+str(self.inputDicts)

if __name__ == '__main__':
    # Tests
    p = Project()
    p.new('test')
    s = Simulation()
